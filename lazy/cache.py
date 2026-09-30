"""KV cache 实现。

KVCache 存每层的 key/value，decode 时复用历史，避免重复算整条序列。
BatchedDecodeCache 是 decode 拼 batch 用的拼接视图，包住各 seq 的
per-seq KVCache。
"""

import torch


class KVCache:
    """单条序列的 KV cache。

    按层存 (key, value)，每步把新算出的 k/v 拼到 seq_len 维上。
    """

    def __init__(self, num_layers: int, device=None) -> None:
        """初始化空 cache。

        Args:
            num_layers: 层数。
            device: cache 所在设备。
        """
        self.num_layers = num_layers
        self.device = device
        self.layer_cache: list[tuple[torch.Tensor, torch.Tensor] |
                               None] = [None for _ in range(num_layers)]
        self._seq_len = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把本步的 key/value 拼进第 layer_idx 层的 cache。

        Args:
            key: 形状 [batch, heads, seq_len_new, head_dim] 的新 key。
            value: 同形状的新 value。
            layer_idx: 层号。

        Returns:
            拼接后的 (key, value)。
        """
        if self.layer_cache[layer_idx] is None:
            self.layer_cache[layer_idx] = (key, value)
        else:
            old_k, old_v = self.layer_cache[layer_idx]
            key_cache = torch.cat([old_k, key], dim=2)
            value_cache = torch.cat([old_v, value], dim=2)
            self.layer_cache[layer_idx] = (key_cache, value_cache)

        self._seq_len = self.layer_cache[layer_idx][0].shape[2]
        return self.layer_cache[layer_idx]

    def get_seq_length(self) -> int:
        """返回当前 cache 里的序列长度。"""
        return self._seq_len

    def reset(self) -> None:
        """清空所有层的 cache。"""
        self.layer_cache = [None for _ in range(self.num_layers)]
        self._seq_len = 0


class BatchedDecodeCache:
    """decode 拼 batch 的"拼接视图"。

    每个 decode step 临时组装：各 seq 历史右 pad 到同长后摆到 batch 维，
    本步新 token 拼在最后一列，update 接口和 KVCache 一样，模型无感。
    pad 列是零值，由 runner 构造的 padding mask 挡掉。

    Attributes:
        caches: 各 seq 的 per-seq KVCache。
        lengths: 各 seq 当前长度（不含本步新 token）。
        max_len: lengths 里的最大值。
    """

    def __init__(self, caches: list[KVCache]) -> None:
        self.caches = caches
        self.lengths = [c.get_seq_length() for c in caches]
        self.max_len = max(self.lengths)

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """拼出本步的 KV 视图，并把新 token 写回 per-seq cache。

        Args:
            key: 形状 [batch, heads, 1, head_dim] 的本步新 key。
            value: 同形状的本步新 value。
            layer_idx: 层号。

        Returns:
            形状 [batch, heads, max_len + 1, head_dim] 的拼接视图，布局是
            [历史 L_i][零 pad][本步新 token]。
        """
        ks, vs = [], []
        for i, cache in enumerate(self.caches):
            k, v = cache.layer_cache[layer_idx]
            pad = self.max_len - k.shape[2]
            if pad:
                k = torch.nn.functional.pad(k, (0, 0, 0, pad))
                v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            ks.append(k)
            vs.append(v)
            # 新 token 当场写回 per-seq cache。
            cache.update(key[i:i + 1], value[i:i + 1], layer_idx)
        k_full = torch.cat([torch.cat(ks, dim=0), key], dim=2)
        v_full = torch.cat([torch.cat(vs, dim=0), value], dim=2)
        return k_full, v_full
