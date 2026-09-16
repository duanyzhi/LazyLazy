# 任务3: KV Cache Manager 多级加载（SSD→CPU→HBM）

## 方案

在任务2 的块粒度 hash prefix cache 之上，把每个缓存块扩展为三级存储：

- **HBM**：块的 KV 张量在显存。
- **CPU**：块的 KV 张量在主机内存。
- **SSD**：块的 KV 张量序列化到磁盘（每块一个 `torch` 文件）。

`lazy/cache_manager.py` 新增 `KVCacheManager(BlockPrefixCache)`，接口与 `BlockPrefixCache` 一致（`match/load/store/clear` + `block_size` + `blocks`），`ModelRunner` 只需把实例替换成它，调用点零改动。

### 容量分配（按 HBM 剩余 buffer）

- `hbm_budget_bytes` 为 `None` 时，预算 = 初始化时刻 `torch.cuda.mem_get_info()` 得到的空闲 HBM 减去 `kv_cache_hbm_reserve_bytes`（默认 1GB 保留），即**用剩余显存来给 KV cache 分配预算**。
- 也可显式指定 `kv_cache_hbm_budget_bytes`（字节）。

### 换入换出（LRU）

- 维护全局 LRU 访问序。
- 新块写入后 `_enforce_budgets()`：HBM 超预算 → 最久未用的 HBM 块逐出到 CPU；CPU 超 `cpu_budget_bytes` → 最久未用的 CPU 块逐出到 SSD（`torch.save`）。
- 命中时 `_ensure_hbm()` 把块逐级搬回：SSD→CPU→HBM（`torch.load(map_location="cpu")` 再 `.cuda()`），或 CPU→HBM。因此**重复命中的请求可以直接从 SSD 加载上来**。

### 多请求 prefix cache

沿用任务2 的块 hash 匹配：多个不同请求各自命中共享块；新块去重后追加，不会被覆盖。

## 关键实现

- `KVCacheManager._ensure_hbm` / `_evict` / `_enforce_budgets` / `tiers()`。
- 每块字节数 = `num_layers * 2 * num_kv_heads * head_dim * block_size * 2`（Qwen3-0.6B + block=4 时约 448KB/块）。
- `match` 继承自 `BlockPrefixCache`，只查 hash 与 token，不关心块当前在哪一级。

## 验证

`tests/test_kv_cache_manager.py`（block=4，HBM=1块、CPU=1块预算，其余溢出到 SSD）：

- 自动预算：`hbm_budget_bytes > 0`（从空闲 HBM 推导）。
- 第 1 次跑 prompt：`tiers = {hbm:1, cpu:1, ssd:5}`，证明块被逐出到 SSD。
- 第 2 次跑同一 prompt：`cached=28`（29 token，7 个整块），输出与第 1 次一致，证明从 SSD 命中并正确加载。
- 第 3 次不同 prompt 共享前缀：`cached=16`（公共前缀 18 token 的整块）。
- 第 4 次再跑原 prompt：`cached=28`，证明多请求共存命中。

`llm_test.py` / `test_prefix_cache.py` / `test_weight_loading.py` 回归通过。

## 范围外

- 未做 SSD 文件的压缩/合并（一块一个文件）。
- 未做并发/异步 prefetch（命中块的加载是同步的）。
- CPU 层未用 pinned memory。
