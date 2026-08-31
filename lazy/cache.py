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

    def get_seq_length(self):
        return self._seq_len

    def reset(self):
        self.layer_cache = [None for _ in range(self.num_layers)]
        self._seq_len = 0


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