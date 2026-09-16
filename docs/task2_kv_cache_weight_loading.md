# 任务2: KV Cache 升级 + 多级权重加载

## 一、KV cache 升级（hash 匹配 + 多请求命中）

任务1 的 prefix cache 是"单条前缀、从第 0 个 token 逐 token 比对、新 prompt 覆盖旧 prefix"。任务2 升级为**块粒度 + hash 匹配 + 多请求共享**。

### 设计

- 把 prompt 按固定 `block_size`（默认 16）切成块。
- 每块的 KV（每层一个 tensor）以**块 token 的稳定 hash** 为 key 存入 `BlockPrefixCache.blocks`。
- 匹配 = 从第 0 块起连续做 dict 查表，命中多少块就复用多少，不再是逐 token 比对。
- 多个不同请求各自命中共有的块；新请求的新块追加进 cache（去重），旧块不会被覆盖，因此**多个请求可以同时命中**。
- 只对整块做匹配/加载，prompt 尾部不足一块的部分始终重算，天然保证"至少重算一个 token"（避免空输入），同时让缓存的 prefix 始终块对齐。

### 关键实现

`lazy/cache.py`：

- `hash_block(token_ids)`：md5 生成的 64-bit 稳定 hash（跨进程稳定，为任务3 SSD 持久化做准备）；entry 内同时存原始 token 用于碰撞校验。
- `BlockPrefixCache`：
  - `match(token_ids)`：返回命中的整块 token 数（块对齐）。
  - `load(kv_cache, token_ids, prefix_len)`：把命中块的 KV `clone()` 拼进运行时 `KVCache`（`prefix_len` 必须块对齐）。
  - `store(token_ids, kv_cache, num_prompt_tokens)`：prefill 后按块切片、去重存入。

`lazy/engine/model_runner.py`：

- `prepare_prefill`：`cached_len = min(matched, n-1)` 后向下取整到块边界，命中块先 `load` 进运行时 cache，`input_ids/positions` 只取后缀。
- `run`：prefill 后 `store(seq.token_ids, kv_cache, seq.num_prompt_tokens)`。

`lazy/config.py` 新增 `prefix_block_size`（默认 16）。

### 验证

`tests/test_prefix_cache.py`（block=16，prompt 18 token）：

- 第 1 次 `cached=0`；第 2 次同 prompt `cached=16` 且输出一致；
- 不同 prompt 共享 16 token 前缀 → `cached=16`；
- 存过 B 之后再跑 A，A 仍能命中 → 证明多请求可命中。

---

## 二、权重多级加载（SSD→CPU→HBM / SSD→HBM）

### 设计

`lazy/utils/loader_weight.py` 新增 `WeightLoader`，管理**解码器层权重**在 HBM(cuda)/CPU RAM/SSD 之间的搬移；`embed_tokens` + 最终 `norm` + rotary buffer 常驻 HBM（小而必需）。

两种模式：

- `three_level`（SSD→CPU→HBM）：先把全部权重从 safetensors 读进 CPU RAM（复用 `load_weights`），forward 时按需把层搬上 HBM，超出预算的层被 LRU 逐回 CPU RAM。
- `two_level`（SSD→HBM）：**不**在 RAM 里保留全量权重。每层按需从 safetensors（mmap 支持）直接读到 HBM，逐出时把参数清成 0 元素占位张量释放显存（shape 从 safetensors 恢复）。面向 nvidia nx 这类统一内存架构，SSD 就是后备存储，省去大块 CPU staging buffer。

控制方式：

- `num_hbm_layers`：同时驻留 HBM 的解码器层数。
- `hbm_budget_bytes`：按字节预算换算 `num_hbm_layers = (budget - 常驻字节) // 单层字节`（至少 1）。

### 前向集成

`lazy/models/qwen3.py` 的 `Qwen3Model.forward` / `Qwen3ForCausalLM.forward` 新增可选 `layer_loader` 回调，在每层前调用 `layer_loader(i)`；`WeightLoader.ensure_layer(i)` 负责把第 i 层搬到 HBM 并按 LRU 逐出超预算的层。

`lazy/engine/model_runner.py` 按 `config.weight_loading_mode` 选择 eager（原行为）或 offload；`run_model` 传入 `layer_loader`。

`lazy/config.py` 新增 `weight_loading_mode` / `num_hbm_layers` / `hbm_budget_bytes`，`LLM/LLMEngine` 透传。

### 验证

`tests/test_weight_loading.py`：同一 prompt + `temperature=0.001`（近似 greedy，确定性），四种配置输出必须一致：

- `eager`（基线）
- `three_level` + `num_hbm_layers=None`（全部驻留）
- `three_level` + `num_hbm_layers=2`（层在 CPU↔HBM 间换入换出）
- `two_level` + `hbm_budget_bytes`（按预算换算为 2 层）

实测全部输出一致；`num_hbm_layers` 解析正确（None→28，2→2，budget→2）。

### 范围外

- 3-level 未用 pinned memory（普通 CPU RAM）。
- 2-level 的"直读"依赖 safetensors 的 mmap + `.to("cuda")` 单张拷贝，未做真正绕过 CPU 页缓存的 `cudaMemcpy`。
- 未做权重压缩/重打包（FreeToken/flash-moe 的重型方案均未引入）。
