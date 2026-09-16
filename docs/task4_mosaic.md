# 任务4: 结合 Mosaic 论文的方案与改动

## 1. 论文核心思想（Mosaic-0916.pdf）

Mosaic（"Bridging the Semantic Gap between LLM Agent Programs and GPU Memory
Management"）指出：LLM Agent 是多轮、多组件的程序，跨调用复用 KV Cache / 权重 /
向量索引，而传统 GPU 内存管理只看「组件本地」的访问历史（LRU/LFU），缺失「程序
语义」，导致两类 mismatch：

1. **Attribution mismatch（对象内）**：组件本地策略按聚合访问历史判断 residency，
   但复用其实属于「仍在运行的特定程序」。典型现象：Agent 在 tool call 期间 KV 块
   空闲，LRU 把它们逐出，下一次该程序回来时被迫重新 prefill（SWE-Bench 上 prefix
   cache 命中率从 96.7% 掉到 39.2%）。
2. **Allocation mismatch（对象间）**：固定 per-object 预算无法跟随多组件需求迁移；
   单纯按当前负载动态调整又忽略了「逐出仍在运行程序将来要复用的数据」的代价。

Mosaic 的解法分三层：

- **AMS（Agent Memory Semantics）**：把内存对象分成三类——program-scoped（KV Cache，
  状态 Running/Suspended/Dead）、shared（权重/索引分区，状态 Active/Idle）、
  transient。运行时由 agent framework 的 dispatch/terminate 事件 + 组件的 unit 上报
  事件更新状态，并产生 per-program use record。
- **LMM（Local Memory Manager）**：做「category-specific lowering」。对 KV Cache：
  - **cache-affinity-driven scheduling**（§6.1.1）：优先调度可复用前缀长的请求，用
    aging credit 防饥饿；用 reservation weight 做准入控制。
  - **liveness-anchored eviction**（§6.1.2）：按「存活程序引用数」逐出（Running 不逐，
    Suspended 保护，Dead 先逐），LRU 作同层 tie-breaker。
- **GMM（Global Memory Manager）**：统一接口收集各 LMM 在候选预算下的 delay estimate
  （service/transition/displacement cost），在线重分配跨对象显存。

## 2. 与 LazyLazy 框架的结合点

LazyLazy 当前已有（任务 1–3）：

| Mosaic 概念 | LazyLazy 现状 | 差距 |
|---|---|---|
| prefix cache（块粒度） | `BlockPrefixCache` 块 hash 匹配 | ✅ 已有 |
| 多级存储 / residency | `KVCacheManager` SSD→CPU→HBM，LRU 换入换出 | ✅ 已有，但**逐出纯 LRU** |
| 权重多级加载（shared data） | `WeightLoader` three/two level | ✅ 已有 |
| **程序语义 / AMS 状态** | 无 | ❌ 缺失（本次补上） |
| **liveness-anchored eviction** | 无（纯 LRU） | ❌ 缺失（本次补上） |
| cache-affinity 调度 / 准入控制 | 无 | 范围外（见 §5） |
| GMM 跨对象分配 | 无 | 范围外（见 §5） |

**结论**：LazyLazy 的 `KVCacheManager._find_lru_tier` 正是论文所批评的「component-
local 策略」（纯 LRU），在 agent 多轮场景下会把「仍存活、将来会复用」的 KV 块逐出到
SSD，导致程序恢复时付出 SSD 重载代价。因此最小、最忠实的结合点就是给 `KVCacheManager`
加上 **AMS 程序状态 + liveness-anchored eviction**。

## 3. 具体改动（`lazy/cache_manager.py`）

- `KVCacheManager.__init__` 新增 `eviction_policy="liveness"`（可选 `"lru"`）。
- 新增程序状态（对应 AMS 的 program-scoped 状态机）：
  - `begin_program(program_id)`
  - `store(..., program_id=None)`：块被某程序使用时标记为 `running`
  - `suspend(program_id)`：`running → suspended`（该程序本次 LLM 调用结束但仍存活）
  - `terminate(program_id)`：`running/suspended → dead`（移除该程序全部状态）
  - `ref_count(h)`：块被多少个存活程序引用
- `_find_lru_tier` 在 `liveness` 策略下：跳过有 `running` 引用的块，按存活引用数
  递增顺序逐出（Dead=0 先逐），LRU 作同层 tie-breaker；等价于论文 §6.1.2。
- 未传 `program_id` / 未调用 suspend/terminate 时，所有块引用数为 0，`liveness`
  退化为纯 LRU，**与任务 1–3 现有行为完全兼容**。

语义与论文一致：一次 `generate` 内 `store` 标记 running → 调用完成后 agent framework
调 `suspend`（程序仍存活）→ 程序结束时调 `terminate`（对应 Mosaic §5.2 的 framework
adapter 四个 hook 中的 dispatch/complete/terminate）。

## 4. 改动前后收益（`tests/test_mosaic_eviction.py`）

场景（block=4，HBM 预算 = 2 个程序的块量，CPU 预算 = 0，超出的块直接落 SSD）：

1. 程序 A：prefill 后 `store(program_id="A")` 再 `suspend("A")`（存活）
2. 程序 B：`store(program_id="B")` 再 `terminate("B")`（已死）
3. 程序 C：新程序 prefill，溢出 HBM 触发逐出

结果：

| 策略 | 程序 A 块在 SSD 的数量 | A 恢复时 load 耗时 |
|---|---|---|
| `lru`（改动前） | 4 / 4（全部被逐到 SSD） | 42.94 ms |
| `liveness`（改动后） | 0 / 4（全部留在 HBM） | 2.88 ms |

- 纯 LRU 把「仍存活、马上要回来」的 A 块当成最久未用逐到 SSD；
- liveness 先逐 Dead 的 B 块，A 块留在 HBM，恢复时从 HBM 直接命中，**约 15× 恢复
  加速**（且两份策略恢复出的 KV 与新鲜 prefill 逐位一致，正确性不损）。

测试脚本 `python tests/test_mosaic_eviction.py` 输出 `PASS: mosaic liveness-anchored
eviction` 即任务 4 停止条件。

## 5. 范围外 / 后续可扩展

- **cache-affinity-driven scheduling + 准入控制**（§6.1.1）：需要 scheduler 感知
  `L(p, C_KV)` 可复用前缀长度与 reservation demand，当前 `Scheduler` 是单请求 prefill
  的最小实现，未接入。
- **γ(p) 保护期衰减**（§6.1.1 式 3）：当前 `suspended` 视为恒定 1 引用（保护期内），
  未实现 `exp(-...)` 衰减。
- **GMM 跨对象分配**（§7）：需要多组件（LLM + embedder + 索引）共置与统一 delay
  estimate 接口，LazyLazy 目前只有 LLM 一个组件。
- **shared data 的 program-informed partition ranking**（§6.2）：需要向量索引组件，
  暂不适用。

若要真正接入 agent framework，只需在其 dispatch/complete/terminate 边界调用上述
`suspend/terminate`，并在 `store` 时传入 `program_id`，与 Mosaic §5.2 的 adapter 模式
一一对应，无需改动 LazyLazy 引擎其余部分。
