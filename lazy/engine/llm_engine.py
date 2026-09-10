from transformers import AutoTokenizer

from lazy.engine.model_runner import ModelRunner
from lazy.config import Config
from lazy.engine.sequence import Sequence
from lazy.engine.scheduler import Scheduler

class LLMEngine:
    def __init__(self, model_path):
        # do config initialization
        config = Config(model_path)

        # do model runner initialization
        self.model_runner = ModelRunner(config)

        # do scheduler initialization
        eos_token_id = config.hf_config.eos_token_id
        if isinstance(eos_token_id, list):  # Qwen3-Base 可能是 list，取第一个
            eos_token_id = eos_token_id[0]
        self.scheduler = Scheduler(config, eos_token_id=eos_token_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

    def add_request(self, prompt: str | list[int], sampling_params):
        # add request to scheduler
        if isinstance(prompt, str):
            # Hi -> [13048]
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # get scheduled sequences from scheduler
        # run model runner
        # postprocess results in scheduler
        seqs, is_prefill = self.scheduler.schedule()

        out_token_id = self.model_runner.run(seqs, is_prefill)

        # log for output tokens
        output_tokens = self.tokenizer.decode(out_token_id)
        print("out token:", output_tokens)
        
        self.scheduler.postprocess(seqs, out_token_id, is_prefill)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts, sampling_params):
        # add requests to scheduler
        # loop until all sequences are finished
        # return final outputs
        self.add_request(prompts, sampling_params)

        while not self.is_finished():
            self.step()

        outputs = []
        for seq in self.scheduler.finished:
            text = self.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens:])
            outputs.append(text)
        self.model_runner.kv_cache.reset()  # 为下一次 generate 清 cache
        self.scheduler.finished.clear()
        return outputs
