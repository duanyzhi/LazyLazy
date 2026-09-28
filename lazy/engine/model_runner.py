import torch

from lazy.models.qwen3 import Qwen3ForCausalLM
from lazy.cache import KVCache
from lazy.config import Config
from lazy.utils.loader_weight import load_weights
from lazy.utils.logger import init_logger
from lazy.engine.sequence import Sequence
from lazy.layers.sampler import Sampler

logger = init_logger(__name__)


class ModelRunner:
    def __init__(self,  config : Config):
        logger.info("building model on cuda (bf16)")
        self.model = Qwen3ForCausalLM(config.hf_config).to(dtype=torch.bfloat16).cuda()
        logger.info("loading weights from %s", config.model)
        assert load_weights(self.model, config.model), "weight missing, please check the model path or hf name"
        self.sampler = Sampler()
        self.kv_cache = KVCache(config.hf_config.num_hidden_layers, device="cuda")
        logger.info("model runner ready")

    def prepare_prefill(self, seqs):
        input_ids = []
        positions = []
        for seq in seqs:
          input_ids.extend(seq.token_ids)
          positions.extend(range(0, len(seq)))

        input_ids = torch.tensor([input_ids], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([positions], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        for seq in seqs:
            input_ids.append(seq.last_token) # decode is get last token
            positions.append(len(seq) - 1)
        input_ids = torch.tensor([input_ids], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([positions], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids, positions

    def prepare_sample(self, seqs):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def run_model(self, input_ids, positions):
        return self.model(input_ids, positions, kv_cache=self.kv_cache)

    def run(self, seqs: Sequence, is_prefill: bool):
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
            logger.debug("prefill: %d seqs, %d tokens", len(seqs), input_ids.shape[1])
        else:
            input_ids, positions = self.prepare_decode(seqs)
            logger.debug("decode: %d seqs", len(seqs))

        temperatures = self.prepare_sample(seqs)

        logits = self.run_model(input_ids, positions)

        token_ids = self.sampler(logits[:, -1, :], temperatures).tolist()
        return token_ids
