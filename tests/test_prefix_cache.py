from lazy import LLM, SamplingParams

MODEL_PATH = "/mnt/mizar/models/Qwen3-0.6B-Base"


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def expected_cached(matched: int, prompt_len: int, block_size: int) -> int:
    """Mirror of ModelRunner.prepare_prefill: leave last token, keep block-aligned."""
    c = min(matched, prompt_len - 1)
    c -= c % block_size
    return c


def main():
    sp = SamplingParams(temperature=0.001, max_tokens=8)
    llm = LLM(MODEL_PATH)
    block = llm.model_runner.prefix_cache.block_size

    pa = "The quick brown fox jumps over the lazy dog near the river bank under the bright morning sun"
    pb = "The quick brown fox jumps over the lazy dog near the river bank under the bright evening moon"

    ta = llm.tokenizer.encode(pa)
    tb = llm.tokenizer.encode(pb)

    full_a = (len(ta) // block) * block  # full blocks of A that can ever be cached

    # 1st run: nothing cached yet
    out1 = llm.generate(pa, sp)
    c1 = llm.model_runner.last_cached_len
    assert c1 == 0, f"first request should cache 0 tokens, got {c1}"

    # 2nd run: full prompt prefix reused
    out2 = llm.generate(pa, sp)
    c2 = llm.model_runner.last_cached_len
    e2 = expected_cached(full_a, len(ta), block)
    assert c2 == e2, f"expected cached {e2}, got {c2}"
    assert c2 > 0, "second request should hit the cache"
    assert out1 == out2, f"outputs differ: {out1!r} vs {out2!r}"

    # 3rd run: different prompt sharing a common prefix (hash-matched blocks)
    out3 = llm.generate(pb, sp)
    c3 = llm.model_runner.last_cached_len
    common = common_prefix_len(ta, tb)
    matched3 = (common // block) * block
    e3 = expected_cached(matched3, len(tb), block)
    assert c3 == e3, f"expected cached {e3}, got {c3}"

    # multiple requests can hit: A is still cacheable after B was stored
    out4 = llm.generate(pa, sp)
    c4 = llm.model_runner.last_cached_len
    assert c4 == e2, f"expected cached {e2}, got {c4}"

    print("PASS: hash prefix cache")
    print(f"  block={block}, lenA={len(ta)}, lenB={len(tb)}, common={common}")
    print(f"  c1={c1}, c2={c2}, c3={c3}, c4={c4}")
    print(f"  cached_blocks={len(llm.model_runner.prefix_cache.blocks)}")


if __name__ == "__main__":
    main()
