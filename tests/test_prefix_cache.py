from lazy import LLM, SamplingParams

MODEL_PATH = "/mnt/mizar/models/Qwen3-0.6B-Base"


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    sampling_params = SamplingParams(temperature=0.001, max_tokens=16)
    llm = LLM(MODEL_PATH)

    prompt = "what is the largest animal in the world?"
    prompt_tokens = llm.tokenizer.encode(prompt)

    # 1st run: no prefix cached yet
    out1 = llm.generate(prompt, sampling_params)
    cached1 = llm.model_runner.last_cached_len
    assert cached1 == 0, f"first request should cache 0 tokens, got {cached1}"

    # 2nd run: full prompt prefix should be reused except the last token
    out2 = llm.generate(prompt, sampling_params)
    cached2 = llm.model_runner.last_cached_len
    assert cached2 == len(prompt_tokens) - 1, (
        f"second request should cache {len(prompt_tokens) - 1} tokens, got {cached2}"
    )

    # greedy-equivalent sampling => identical outputs
    assert out1 == out2, f"outputs differ: {out1!r} vs {out2!r}"

    # 3rd run: different prompt sharing a common prefix
    prompt3 = "what is the largest animal in the ocean?"
    tokens3 = llm.tokenizer.encode(prompt3)
    out3 = llm.generate(prompt3, sampling_params)
    cached3 = llm.model_runner.last_cached_len
    # Cache the full matched prefix; the "leave last token" rule only applies
    # when the whole prompt matches (matched == num_prompt_tokens).
    common = common_prefix_len(prompt_tokens, tokens3)
    expected3 = min(common, len(tokens3) - 1)
    assert cached3 == expected3, f"expected cached {expected3}, got {cached3}"

    print("PASS: prefix cache MVP")
    print(f"  prompt_tokens={len(prompt_tokens)}")
    print(f"  cached1={cached1}, cached2={cached2}, cached3={cached3}")
    print(f"  out2={out2[0]!r}")


if __name__ == "__main__":
    main()
