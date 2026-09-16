# Claude 工作日志 (Worklog)

本文件记录 Claude 在 LazyLazy 项目中每一步的关键节点与中间结论，用于断点续跑。

## 会话概览

- 开始时间: 2026-09-16
- 分支: update_kv_cache
- 基线提交: ff4ce9d (run succeed for decode)

## 任务清单

1. 任务1: 按 skills/0916.md 实现 prefix cache MVP，测试通过 → commit + 文档
2. 任务2: 升级 KV cache（hash 匹配、多请求命中、SSD->CPU->HBM 三级 / SSD->HBM 两级权重加载）
3. 任务3: KV cache manager 支持 SSD->CPU->HBM 多级加载，按 HBM 剩余 buffer 分配，多请求 prefix cache，重复命中从 SSD 加载
4. 任务4: 阅读 Mosaic-0916.pdf，输出结合方案文档，尝试改动并测试收益

## 基线环境确认

- GPU: NVIDIA A100-SXM4-80GB, 空显存
- venv: /mnt/duanyunzhi/roads/.venv
- 模型: /mnt/mizar/models/Qwen3-0.6B-Base (Qwen3-0.6B, 28 层)
- 基线测试 `python tests/llm_test.py` 通过，输出正常

---

## 步骤记录

### [STEP 0] 环境与基线确认
- 结论: 已通读 0916.md、lazy 包全部代码；基线测试通过。
- 关键文件:
  - `lazy/cache.py` (KVCache, 62 行)
  - `lazy/engine/model_runner.py` (ModelRunner, 57 行)
  - `lazy/engine/llm_engine.py` (LLMEngine.generate)
  - `lazy/engine/scheduler.py`, `lazy/engine/sequence.py`
  - `lazy/models/qwen3.py` (attention 已支持 q_len != kv_len 的 causal mask)
- 0916.md 已有明确 MVP 实现方案（直接照做）。

### [STEP 1] 任务1 完成 (commit 54b4dd3)
- 结论: prefix cache MVP 开发 + 测试通过。
- 改动:
  - `lazy/cache.py`: KVCache.set_layer + PrefixCache 类
  - `lazy/engine/model_runner.py`: prepare_prefill 复用前缀、run 后 store、last_cached_len 供测试
  - 新增 `tests/test_prefix_cache.py`、`docs/prefix_cache_mvp.md`
- 测试结果: prompt_tokens=9 → cached1=0, cached2=8, cached3=7；llm_test.py 回归通过。
- 关键经验: 测试里 expected3 公式应为 `min(公共前缀, prompt_len-1)`，不是 `公共前缀-1`（"留最后一个 token"规则只在整段 prompt 命中时生效）。

### [STEP 2] 任务2 完成
- 结论: KV cache 升级(hash+block+多请求) + 权重多级加载(3级/2级) 开发 + 测试通过。
- 改动:
  - `lazy/cache.py`: hash_block + BlockPrefixCache(替换 PrefixCache)
  - `lazy/engine/model_runner.py`: 块对齐匹配 + WeightLoader 集成
  - `lazy/utils/loader_weight.py`: 新增 WeightLoader(three_level/two_level)
  - `lazy/models/qwen3.py`: forward 加 layer_loader 回调
  - `lazy/config.py`: prefix_block_size / weight_loading_mode / num_hbm_layers / hbm_budget_bytes
  - `lazy/engine/llm_engine.py`: 透传权重加载参数
  - 新增 `tests/test_weight_loading.py`, 更新 `tests/test_prefix_cache.py`, `docs/task2_kv_cache_weight_loading.md`
- 测试结果: test_prefix_cache 通过(c1=0,c2=16,c3=16,c4=16); test_weight_loading 通过(4种模式输出一致); llm_test 回归通过。
- 关键经验: `param.data = torch.empty(..., device="meta")` 会报 "incompatible tensor type"，meta 张量不能赋给 Parameter.data；用 `torch.empty(0, dtype=param.dtype, device="cpu")` 释放存储（shape 从 safetensors 恢复）。
- 关键数据: Qwen3-0.6B embed+norm 常驻 ≈ 311MB，单 decoder 层 ≈ 31.5MB(budget 换算用)。

### [STEP 3] 任务3 完成
- 结论: KV cache manager SSD->CPU->HBM 多级加载 开发 + 测试通过。
- 改动:
  - 新增 `lazy/cache_manager.py`: KVCacheManager(BlockPrefixCache) 三级存储 + LRU 换入换出
  - `lazy/engine/model_runner.py`: prefix_cache 换成 KVCacheManager
  - `lazy/config.py` / `lazy/engine/llm_engine.py`: 新增 kv_cache_* 与 prefix_block_size 参数
  - 新增 `tests/test_kv_cache_manager.py`, `docs/task3_kv_cache_manager.md`
- 测试结果: 自动预算>0; run1 tiers={hbm:1,cpu:1,ssd:5}; c1=0,c2=28,c3=16,c4=28; 全量回归通过。
- 关键数据: Qwen3-0.6B block=4 时每块 458752 字节。



