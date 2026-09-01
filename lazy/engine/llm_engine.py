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
        self.scheduler = Scheduler(config)
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
        pass

    def generate(self, prompts, sampling_params):
        # add requests to scheduler
        # loop until all sequences are finished
        # return final outputs
        self.add_request(prompts, sampling_params)

        seq, is_prefill = self.scheduler.schedule()
        print(seq, is_prefill)

        out_token_id = self.model_runner.run(seq, is_prefill)

        output_tokens = self.tokenizer.decode(out_token_id)

        print("out token:", output_tokens)
