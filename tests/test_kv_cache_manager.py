import tempfile

from transformers import AutoConfig

from lazy import LLM, SamplingParams

MODEL_PATH = "/mnt/mizar/models/Qwen3-0.6B-Base"
PROMPT = ("The quick brown fox jumps over the lazy dog near the river bank under the "
          "bright morning sun and watches the fish swim by the tall green reeds")
PROMPT2 = ("The quick brown fox jumps over the lazy dog near the river bank under the "
           "bright morning sun but then it starts to rain heavily")


def expected_cached(matched: int, prompt_len: int, block_size: int) -> int:
    c = min(matched, prompt_len - 1)
    return c - c % block_size


def block_bytes(block_size: int) -> int:
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * block_size * 2


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    sp = SamplingParams(temperature=0.001, max_tokens=4)
    bs = 4
    bb = block_bytes(bs)

    # auto budget derived from remaining HBM
    llm_auto = LLM(MODEL_PATH, prefix_block_size=bs)
    auto_budget = llm_auto.model_runner.prefix_cache.hbm_budget_bytes
    assert auto_budget > 0
    print(f"auto hbm budget bytes (from free HBM): {auto_budget}")

    # tiny HBM + CPU budget (1 block each) so blocks overflow to SSD
    ssd = tempfile.mkdtemp(prefix="lazy_kv_")
    llm = LLM(
        MODEL_PATH,
        prefix_block_size=bs,
        kv_cache_hbm_budget_bytes=bb,
        kv_cache_cpu_budget_bytes=bb,
        kv_cache_ssd_dir=ssd,
    )

    ta = llm.tokenizer.encode(PROMPT)
    tb = llm.tokenizer.encode(PROMPT2)
    full_a = (len(ta) // bs) * bs

    # 1st run: store blocks, some must be evicted to SSD (capacity = 1 hbm + 1 cpu)
    out1 = llm.generate(PROMPT, sp)
    c1 = llm.model_runner.last_cached_len
    assert c1 == 0, c1
    tiers1 = llm.model_runner.prefix_cache.tiers()
    print("after run1 tiers:", tiers1)
    assert tiers1["ssd"] > 0, f"expected blocks evicted to SSD, got {tiers1}"
    assert tiers1["hbm"] <= 1, f"hbm should hold <= 1 block, got {tiers1}"

    # 2nd run: repeated hit served from SSD (loaded back SSD->CPU->HBM)
    out2 = llm.generate(PROMPT, sp)
    c2 = llm.model_runner.last_cached_len
    e2 = expected_cached(full_a, len(ta), bs)
    assert c2 == e2, f"expected cached {e2}, got {c2}"
    assert out1 == out2, f"outputs differ: {out1!r} vs {out2!r}"

    # 3rd run: different prompt sharing a prefix (multi-request)
    out3 = llm.generate(PROMPT2, sp)
    c3 = llm.model_runner.last_cached_len
    common = common_prefix_len(ta, tb)
    e3 = expected_cached((common // bs) * bs, len(tb), bs)
    assert c3 == e3, f"expected cached {e3}, got {c3}"

    # 4th run: original prompt still cacheable after another request
    out4 = llm.generate(PROMPT, sp)
    c4 = llm.model_runner.last_cached_len
    assert c4 == e2, f"expected cached {e2}, got {c4}"

    print("PASS: kv cache manager")
    print(f"  block={bs}, block_bytes={bb}, lenA={len(ta)}, lenB={len(tb)}, common={common}")
    print(f"  c1={c1}, c2={c2}, c3={c3}, c4={c4}")
    print(f"  final tiers={llm.model_runner.prefix_cache.tiers()}")


if __name__ == "__main__":
    main()
