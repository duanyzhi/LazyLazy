import time

from transformers import AutoTokenizer

from lazy.engine.model_runner import ModelRunner
from lazy.config import Config
from lazy.engine.sequence import Sequence
from lazy.engine.scheduler import Scheduler
from lazy.utils.logger import init_logger

logger = init_logger(__name__)


class LLMEngine:
    def __init__(self, model_path):
        logger.info("initializing engine, model: %s", model_path)
        # do config initialization
        config = Config(model_path)

        # do model runner initialization
        logger.info("loading model weights (this may take a while)")
        self.model_runner = ModelRunner(config)

        # do scheduler initialization
        eos_token_id = config.hf_config.eos_token_id
        if isinstance(eos_token_id, list):  # Qwen3-Base 可能是 list，取第一个
            eos_token_id = eos_token_id[0]
        self.scheduler = Scheduler(config, eos_token_id=eos_token_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        logger.info("engine ready")

    def add_request(self, prompt: str | list[int], sampling_params):
        # add request to scheduler
        if isinstance(prompt, str):
            # Hi -> [13048]
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        logger.info("request added: seq %d, %d prompt tokens", seq.seq_id, seq.num_prompt_tokens)

    def step(self):
        # get scheduled sequences from scheduler
        # run model runner
        # postprocess results in scheduler
        seqs, is_prefill = self.scheduler.schedule()

        out_token_id = self.model_runner.run(seqs, is_prefill)

        logger.debug("step: %d seqs, is_prefill=%s, out token: %s",
                     len(seqs), is_prefill, self.tokenizer.decode(out_token_id))

        self.scheduler.postprocess(seqs, out_token_id, is_prefill)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts, sampling_params):
        # add requests to scheduler
        # loop until all sequences are finished
        # return final outputs
        self.add_request(prompts, sampling_params)
        start_time = time.perf_counter()

        while not self.is_finished():
            self.step()

        outputs = []
        for seq in self.scheduler.finished:
            text = self.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens:])
            outputs.append(text)
            num_new_tokens = seq.num_tokens - seq.num_prompt_tokens
            elapsed = time.perf_counter() - start_time
            logger.info("generate done: seq %d, %d tokens in %.2fs (%.2f tok/s)",
                        seq.seq_id, num_new_tokens, elapsed, num_new_tokens / max(elapsed, 1e-9))
        self.model_runner.kv_cache.reset()  # 为下一次 generate 清 cache
        self.scheduler.finished.clear()
        return outputs
