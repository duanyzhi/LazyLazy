from transformers import AutoTokenizer

from lazy.engine.model_runner import ModelRunner
from lazy.config import Config


class LLMEngine:
    def __init__(self, model_path):
        config = Config(model_path)
        self.model_runner = ModelRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # do model runner initialization
        # do config initialization
        # do scheduler initialization
        pass

    def add_request(self, prompt: str | list[int], sampling_params):
        # add request to scheduler
        pass

    def step(self):
        # get scheduled sequences from scheduler
        # run model runner
        # postprocess results in scheduler
        pass

    def generate(self, prompts, sampling_params):
        # add requests to scheduler
        # loop until all sequences are finished
        # return final outputs
        pass
