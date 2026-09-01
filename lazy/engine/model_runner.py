import torch

from lazy.models.qwen3 import Qwen3ForCausalLM
from lazy.config import Config
from lazy.utils.loader_weight import load_weights
from lazy.engine.sequence import Sequence
from lazy.layers.sampler import Sampler

class ModelRunner:
    def __init__(self,  config : Config):
        self.model = Qwen3ForCausalLM(config.hf_config).to(dtype=torch.bfloat16).cuda()
        assert load_weights(self.model, config.model), "weight missing, please check the model path or hf name"
        self.sampler = Sampler()

    def prepare_prefill(self, seqs):
        input_ids = []
        positions = []
        for seq in seqs:
          input_ids.extend(seq.token_ids)
          positions.extend(range(0, len(seq)))

        input_ids = torch.tensor([input_ids], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([positions], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids, positions

    def prepare_decode(self, seqs):
        pass

    def prepare_sample(self, seqs):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def run_model(self, input_ids, positions, is_prefill: bool):
        logits = self.model(input_ids, positions)
        return logits

    def run(self, seqs: Sequence, is_prefill: bool):
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        temperatures = self.prepare_sample(seqs)

        logits = self.run_model(input_ids, positions, is_prefill)

        token_ids = self.sampler(logits[:, -1, :], temperatures).tolist()
        return token_ids
