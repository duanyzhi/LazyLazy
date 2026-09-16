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
