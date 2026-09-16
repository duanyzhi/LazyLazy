import hashlib

import torch

"""
KVCache is a simple key-value cache for storing the key and value tensors for each layer of a transformer model. 
It is used to store the past key and value tensors for each layer during autoregressive generation, 
so that they can be reused in subsequent forward passes. 
This avoids recomputing the key and value tensors for the entire sequence, which can be expensive for long sequences.
"""
class KVCache:
    def __init__(self, num_layers: int, device=None):
        self.num_layers = num_layers
        self.device = device
        self.layer_cache = [None for _ in range(num_layers)]
        self._seq_len = 0

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        """
        key/value shape: [batch, heads, seq_len_new, head_dim]
        """
        if self.layer_cache[layer_idx] is None:
            self.layer_cache[layer_idx] = (key, value)
        else:
            old_k, old_v = self.layer_cache[layer_idx]
            key_cache = torch.cat([old_k, key], dim=2)  # concatenate along seq_len dimension
            value_cache = torch.cat([old_v, value], dim=2)
            self.layer_cache[layer_idx] = (key_cache, value_cache)

        self._seq_len = self.layer_cache[layer_idx][0].shape[2]
        return self.layer_cache[layer_idx]

    def set_layer(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        self.layer_cache[layer_idx] = (key, value)
        self._seq_len = key.shape[2]

    def get_seq_length(self):
        return self._seq_len

    def reset(self):
        self.layer_cache = [None for _ in range(self.num_layers)]
        self._seq_len = 0


def hash_block(token_ids: list[int]) -> int:
    """Stable 64-bit hash of a token block (consistent across processes).

    md5 keeps the key stable even across Python's per-process hash seed, which is
    required once blocks are persisted to SSD (task 3). The raw token tuple is also
    stored in each entry so a hash collision can never silently return wrong KV.
    """
    data = repr(list(token_ids)).encode("utf-8")
    return int.from_bytes(hashlib.md5(data).digest()[:8], "little")


class BlockPrefixCache:
    """Block-granular, hash-matched prefix cache shared across requests.

    The prompt is split into fixed-size blocks of `block_size` tokens. Each block's
    KV (one tensor per layer) is stored under the hash of the block's token ids, so
    matching is a dict lookup instead of a token-by-token scan, and many distinct
    requests can each hit their own cached blocks (only the common blocks are shared).

    Only full blocks participate in matching/loading. The tail partial block of a
    prompt is always recomputed, which also guarantees at least one token is
    recomputed even when the whole prompt is cached.
    """

    def __init__(self, num_layers: int, block_size: int = 16, device=None):
        self.num_layers = num_layers
        self.block_size = block_size
        self.device = device
        self.blocks: dict[int, dict] = {}  # hash -> {"token_ids": tuple, "kv": [(k, v) per layer]}

    def match(self, token_ids: list[int]) -> int:
        matched = 0
        n = len(token_ids)
        for start in range(0, n - n % self.block_size, self.block_size):
            block = token_ids[start:start + self.block_size]
            entry = self.blocks.get(hash_block(block))
            if entry is None or entry["token_ids"] != tuple(block):
                break
            matched += self.block_size
        return matched

    def load(self, kv_cache: KVCache, token_ids: list[int], prefix_len: int):
        if prefix_len <= 0:
            return
        assert prefix_len % self.block_size == 0, "prefix_len must be block-aligned"
        for layer_idx in range(self.num_layers):
            k_chunks = []
            v_chunks = []
            for start in range(0, prefix_len, self.block_size):
                block = token_ids[start:start + self.block_size]
                k, v = self.blocks[hash_block(block)]["kv"][layer_idx]
                k_chunks.append(k.clone())
                v_chunks.append(v.clone())
            kv_cache.set_layer(layer_idx, torch.cat(k_chunks, dim=2), torch.cat(v_chunks, dim=2))

    def store(self, token_ids: list[int], kv_cache: KVCache, num_prompt_tokens: int):
        for start in range(0, num_prompt_tokens - num_prompt_tokens % self.block_size, self.block_size):
            block = token_ids[start:start + self.block_size]
            h = hash_block(block)
            if h in self.blocks:
                continue
            entry = {"token_ids": tuple(block), "kv": []}
            for layer_idx in range(self.num_layers):
                k, v = kv_cache.layer_cache[layer_idx]
                entry["kv"].append(
                    (k[:, :, start:start + self.block_size].clone(),
                     v[:, :, start:start + self.block_size].clone())
                )
            self.blocks[h] = entry

    def clear(self):
        self.blocks = {}


# def create_causal_mask(
#     attention_mask: torch.Tensor,
#     past_seq_len: int = 0,
#     device=None,
# ):
#     """
#     attention_mask: [batch, seq_len], values are 0/1
#     seq_len: length of the current input sequence, prefix length is input prompt len, for decode seq_len = 1
#     past_seq_len: history length of the past key/value cache
#     """
#     batch, seq_len = attention_mask.shape
#     total_len = past_seq_len + seq_len # all len for current input to attention

#     # build causal mask on the current block
#     q = seq_len
#     k = total_len
#     query_pos = torch.arange(q, device=device).unsqueeze(1)
#     key_pos = torch.arange(k, device=device).unsqueeze(0)

#     valid = key_pos <= (past_seq_len + query_pos)

#     # pad positions are invalid
#     pad_mask = attention_mask[:, :seq_len].unsqueeze(1)  # [b,1,seq_len] 还需再变换
#     # 这里通常是更完整的实现，下面这个版本先保留最小版
#     return valid.unsqueeze(0).unsqueeze(0)