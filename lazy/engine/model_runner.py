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
        self.num_layers = config.hf_config.num_hidden_layers
        self.kv_caches: dict[int, KVCache] = {}  # seq_id -> 每个请求独立的 KV cache
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

    def run_model(self, input_ids, positions, kv_cache):
        return self.model(input_ids, positions, kv_cache=kv_cache)

    def free_cache(self, seq_id: int):
        self.kv_caches.pop(seq_id, None)

    def _run_seq(self, seq: Sequence, is_prefill: bool):
        if is_prefill:
            self.kv_caches[seq.seq_id] = KVCache(self.num_layers, device="cuda")
            input_ids, positions = self.prepare_prefill([seq])
            logger.debug("prefill: seq %d, %d tokens", seq.seq_id, input_ids.shape[1])
        else:
            input_ids, positions = self.prepare_decode([seq])
        temperatures = self.prepare_sample([seq])
        logits = self.run_model(input_ids, positions, self.kv_caches[seq.seq_id])
        return self.sampler(logits[:, -1, :], temperatures).item()

    def run(self, seqs: list[Sequence], is_prefill: bool):
        if not is_prefill:
            logger.debug("decode: %d seqs", len(seqs))
        return [self._run_seq(seq, is_prefill) for seq in seqs]
