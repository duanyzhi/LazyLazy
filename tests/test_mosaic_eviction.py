import tempfile
import time

import torch

from lazy.cache import KVCache, hash_block
from lazy.cache_manager import KVCacheManager
from lazy.config import Config
from lazy.engine.model_runner import ModelRunner

MODEL_PATH = "/mnt/mizar/models/Qwen3-0.6B-Base"
BS = 4  # block size

# Three distinct, block-aligned prompts (raw token ids, all < vocab size 151936).
A = list(range(1, 17))    # 16 tokens -> 4 blocks
B = list(range(101, 117))
C = list(range(201, 217))
N_BLOCKS = len(A) // BS


def prefill(runner, token_ids):
    """Run one prefill forward pass and leave its KV in runner.kv_cache."""
    runner.kv_cache.reset()
    input_ids = torch.tensor([token_ids], dtype=torch.int64).cuda()
    positions = torch.tensor([list(range(len(token_ids)))], dtype=torch.int64).cuda()
    with torch.no_grad():
        runner.model(input_ids, positions, kv_cache=runner.kv_cache)
    return runner.kv_cache


def ssd_blocks(pm, tokens):
    """How many full blocks of `tokens` currently live in the SSD tier."""
    return sum(
        1 for start in range(0, len(tokens), BS)
        if pm.blocks[hash_block(tokens[start:start + BS])]["tier"] == "ssd"
    )


def main():
    config = Config(MODEL_PATH)
    runner = ModelRunner(config)
    hf = config.hf_config
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    block_bytes = hf.num_hidden_layers * 2 * hf.num_key_value_heads * head_dim * BS * 2
    # HBM holds two programs' blocks (A+B); the third program (C) forces eviction
    # of exactly one program's worth of blocks.
    hbm_budget = 2 * N_BLOCKS * block_bytes

    def scenario(policy, ssd_dir):
        pm = KVCacheManager(
            hf.num_hidden_layers, block_size=BS, device="cuda",
            hbm_budget_bytes=hbm_budget, cpu_budget_bytes=0,
            ssd_dir=ssd_dir, eviction_policy=policy,
        )
        # program A: live (its turn finished, may come back on a later turn)
        prefill(runner, A)
        pm.store(A, runner.kv_cache, len(A), program_id="A")
        pm.suspend("A")
        # program B: terminated (its blocks are dead)
        prefill(runner, B)
        pm.store(B, runner.kv_cache, len(B), program_id="B")
        pm.terminate("B")
        # program C: a new program whose blocks overflow HBM and trigger eviction
        prefill(runner, C)
        pm.store(C, runner.kv_cache, len(C), program_id="C")
        return pm

    pm_lru = scenario("lru", tempfile.mkdtemp(prefix="lazy_lru_"))
    pm_live = scenario("liveness", tempfile.mkdtemp(prefix="lazy_live_"))

    ssd_lru_A = ssd_blocks(pm_lru, A)
    ssd_live_A = ssd_blocks(pm_live, A)
    ssd_live_B = ssd_blocks(pm_live, B)
    print(f"program A blocks in SSD: lru={ssd_lru_A}, liveness={ssd_live_A}")
    print(f"program B (dead) blocks in SSD under liveness: {ssd_live_B}")

    # The benefit: liveness-anchored eviction keeps the live program's blocks in
    # HBM instead of demoting them to SSD like pure LRU does.
    assert ssd_lru_A == N_BLOCKS, f"LRU should evict A's {N_BLOCKS} blocks to SSD, got {ssd_lru_A}"
    assert ssd_live_A == 0, f"liveness should keep A's blocks in HBM, got {ssd_live_A} in SSD"
    assert ssd_live_B == N_BLOCKS, f"liveness should evict dead B blocks first, got {ssd_live_B}"

    # Both policies still serve the same (correct) KV on resume; only the tier differs.
    prefill(runner, A)
    ref_k0 = runner.kv_cache.layer_cache[0][0].clone()
    for name, pm in (("lru", pm_lru), ("liveness", pm_live)):
        kv = KVCache(hf.num_hidden_layers, device="cuda")
        t0 = time.perf_counter()
        pm.load(kv, A, N_BLOCKS * BS)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        assert torch.equal(kv.layer_cache[0][0], ref_k0), f"{name} load returned wrong KV"
        print(f"{name} resume load time: {dt * 1e3:.2f} ms")

    print("PASS: mosaic liveness-anchored eviction")
    print(f"  blocks={N_BLOCKS}, block_bytes={block_bytes}, hbm_budget={hbm_budget}")
    print(f"  A(ssd): lru={ssd_lru_A}, liveness={ssd_live_A}")


if __name__ == "__main__":
    main()
