import torch

from lazy.models.qwen3 import Qwen3ForCausalLM
from lazy.cache import KVCache, PrefixCache
from lazy.config import Config
from lazy.utils.loader_weight import load_weights
from lazy.engine.sequence import Sequence
from lazy.layers.sampler import Sampler

class ModelRunner:
    def __init__(self,  config : Config):
        self.model = Qwen3ForCausalLM(config.hf_config).to(dtype=torch.bfloat16).cuda()
        assert load_weights(self.model, config.model), "weight missing, please check the model path or hf name"
        self.sampler = Sampler()
        self.kv_cache = KVCache(config.hf_config.num_hidden_layers, device="cuda")
        self.prefix_cache = PrefixCache(config.hf_config.num_hidden_layers, device="cuda")
        self.last_cached_len = 0

    def prepare_prefill(self, seqs):
        if len(seqs) != 1:
            raise NotImplementedError("prefix cache MVP only supports one prefill seq")

        seq = seqs[0]
        self.kv_cache.reset()

        matched = self.prefix_cache.match(seq.token_ids)
        cached_len = min(matched, seq.num_prompt_tokens - 1)
        self.last_cached_len = cached_len

        seq.num_cached_tokens = cached_len
        seq.num_scheduled_tokens = seq.num_prompt_tokens - cached_len

        if cached_len > 0:
            self.prefix_cache.load_prefix(self.kv_cache, cached_len)

        input_ids = seq.token_ids[cached_len:]
        positions = list(range(cached_len, seq.num_prompt_tokens))

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
        else:
            input_ids, positions = self.prepare_decode(seqs)

        temperatures = self.prepare_sample(seqs)

        logits = self.run_model(input_ids, positions)

        if is_prefill:
            self.prefix_cache.store(seqs[0].token_ids, self.kv_cache)

        token_ids = self.sampler(logits[:, -1, :], temperatures).tolist()
        return token_ids
