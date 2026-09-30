import torch
import torch.nn as nn
from transformers import Qwen3Config

from lazy.cache import KVCache
from lazy.layers.norm import RMSNorm
from lazy.layers.rotary_embedding import get_rope


class SiLU(nn.Module):
    """SiLU 激活函数。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(x)


class Qwen3MLP(nn.Module):
    """Qwen3 的 gated MLP。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size,
                                   self.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(self.hidden_size,
                                 self.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,
                                   self.hidden_size,
                                   bias=False)
        self.act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维对半切开再换位，RoPE 用。"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 rotary 位置编码加到 query 和 key 上。

    Args:
        q: query 张量。
        k: key 张量。
        cos: rotary 的 cos 部分。
        sin: rotary 的 sin 部分。
        unsqueeze_dim: 给 cos/sin 插入的维度，默认 1，方便和 q/k 广播。

    Returns:
        加了位置编码的 (query, key)。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头复制 n_rep 份，对齐 query 的头数。

    等价于 torch.repeat_interleave(x, dim=1, repeats=n_rep)，把形状从
    (batch, num_key_value_heads, seqlen, head_dim) 变成
    (batch, num_attention_heads, seqlen, head_dim)。
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """eager 版 attention。

    Args:
        module: attention 模块，提供 num_key_value_groups / scaling。
        query: 形状 [B, heads, q_len, head_dim]。
        key: 形状 [B, kv_heads, kv_len, head_dim]。
        value: 同 key 的形状。
        attention_mask: decode 拼 batch 的 padding mask，
            [B, 1, 1, kv_len]，True 表示挡住。

    Returns:
        (attention 输出, attention 权重)。
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2,
                                                            3)) * module.scaling

    if module.is_causal:
        q_len, kv_len = query.shape[-2], key_states.shape[-2]
        # query 第 i 个位置对应整条序列的第 (kv_len - q_len + i) 个 token。
        q_pos = torch.arange(kv_len - q_len, kv_len,
                             device=query.device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=query.device).unsqueeze(0)
        causal_mask = k_pos > q_pos
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    if attention_mask is not None:
        attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))

    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Qwen3 的多头注意力。"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        if hasattr(config, "layer_types"):
            self.layer_type = config.layer_types[layer_idx]
        else:
            self.layer_type = None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = (config.num_attention_heads //
                                     config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position = getattr(config, "max_position_embeddings", 2048)

        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", 10000)
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

        self.q_proj = nn.Linear(config.hidden_size,
                                config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim,
                                config.hidden_size,
                                bias=config.attention_bias)
        # q_norm/k_norm 只作用在 head_dim 上，所以 q_norm 之后不用再 reshape。
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        if self.layer_type == "sliding_attention":
            self.sliding_window = config.sliding_window
        else:
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q, k = self.rotary_emb(positions, q, k)

        if kv_cache is not None:
            # prefill 原样存入，decode 拼接历史。
            k, v = kv_cache.update(k, v, self.layer_idx)

        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v, attention_mask=attention_mask)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 的一层 decoder。"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        residual: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(
                hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states, _ = self.self_attn(hidden_states, position_ids, kv_cache,
                                          attention_mask)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """embedding + 若干 decoder 层 + 最后一层 norm。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """逐层跑 forward。

        Args:
            input_ids: [batch, seq_len] 的 token id。
            position_ids: [batch, seq_len] 的位置。
            kv_cache: 本序列的 KV cache。
            attention_mask: decode 拼 batch 的 padding mask。

        Returns:
            最后一层的 hidden states。
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states=hidden_states,
                                            position_ids=position_ids,
                                            residual=residual,
                                            kv_cache=kv_cache,
                                            attention_mask=attention_mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """带 lm_head 的 Qwen3，输出每个位置的 logits。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """跑模型并算 logits。

        Args:
            input_ids: [batch, seq_len] 的 token id。
            position_ids: [batch, seq_len] 的位置。
            kv_cache: 本序列的 KV cache。
            attention_mask: decode 拼 batch 的 padding mask。

        Returns:
            [batch, seq_len, vocab_size] 的 logits。
        """
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(outputs)
        return logits
