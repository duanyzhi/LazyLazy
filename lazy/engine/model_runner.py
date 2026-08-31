from lazy.models.qwen3 import Qwen3ForCausalLM
from lazy.config import Config

class ModelRunner:
    def __init__(self,  config : Config):
        self.model = Qwen3ForCausalLM(config.hf_config)

    def prepare_prefill(self, seqs):
        pass

    def prepare_decode(self, seqs):
        pass

    def prepare_sample(self, seqs):
        pass

    def run_model(self, input_ids, positions, is_prefill: bool):
        logits = self.model(input_ids, positions)
        return logits

    def sampler(self, logits, temperatures):
        pass

    def run(self, seqs, is_prefill: bool):
        pass
        # input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # temperatures = self.prepare_sample(seqs)
        # logits = self.run_model(input_ids, positions, is_prefill)
        # token_ids = self.sampler(logits, temperatures).tolist()
        # return token_ids
