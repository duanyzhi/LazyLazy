# Prefix Cache MVP (任务1)

跨请求复用的连续前缀缓存 MVP。实现与 `skills/0916.md` 方案一致，代码改动集中在 `lazy/cache.py` 与 `lazy/engine/model_runner.py`。

## 改动

### `lazy/cache.py`

- `KVCache` 新增 `set_layer(layer_idx, key, value)`：把现成的前缀 KV 直接塞进运行时缓存。
- 新增 `PrefixCache`：
  - `match(token_ids)`：从第 0 个 token 起连续比对，返回公共前缀长度。
  - `load_prefix(kv_cache, prefix_len)`：把缓存的前 `prefix_len` 个 token 的 KV `clone()` 进运行时缓存（必须 clone，避免与运行时缓存共享显存）。
  - `store(token_ids, kv_cache)`：prefill 结束后克隆整段 prompt KV。
  - `clear()`：清空。

### `lazy/engine/model_runner.py`

- `__init__` 新建 `self.prefix_cache = PrefixCache(...)`，并记录 `self.last_cached_len`（供测试断言）。
- `prepare_prefill`：
  - 只支持单 prefill seq，多 seq 抛 `NotImplementedError`。
  - 先 `kv_cache.reset()`，再 `prefix_cache.match()`。
  - `cached_len = min(matched, num_prompt_tokens - 1)`：整段 prompt 命中时仍保留最后一个 token 真正跑前向，避免空输入。
  - 命中前缀先 `load_prefix` 进运行时 KV，`input_ids/positions` 只取后缀，`positions` 从 `cached_len` 起（保证 RoPE/因果 mask 绝对位置正确）。
- `run`：prefill 前向结束后、postprocess 之前调用 `prefix_cache.store(seq.token_ids, kv_cache)`，此时 `seq.token_ids` 仍是纯 prompt，不会把生成 token 存进去。

## 关键细节

- prefix cache 与运行时 KV 分离，`generate()` 结尾的 `kv_cache.reset()` 只清运行时 KV，不清 prefix。
- attention 因果 mask 已支持 `q_len != kv_len`（`eager_attention_forward` 按 `kv_len - q_len` 对齐绝对位置），无需改 attention。
- 命中的前缀 KV 已带正确 RoPE；后缀按 `cached_len` 起的绝对位置重新算 RoPE。

## 验证

测试脚本：`tests/test_prefix_cache.py`

- 同一 prompt 跑两次：第一次 `cached=0`，第二次 `cached = prompt_len - 1`，且两次输出一致。
- 不同 prompt 共享前缀：`cached = min(公共前缀长度, prompt_len - 1)`。

实测结果（Qwen3-0.6B-Base，prompt 9 token）：

```
prompt_tokens=9
cached1=0, cached2=8, cached3=7
```

`tests/llm_test.py` 回归通过。

## 范围外

- 多请求 batch、多 prefix 同时保留。
- block table、分块哈希。
- 缓存淘汰、显存水位控制。

以上由任务 2/3 承接。
