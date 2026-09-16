from lazy import LLM, SamplingParams

MODEL_PATH = "/mnt/mizar/models/Qwen3-0.6B-Base"
PROMPT = "what is the largest animal in the world?"


def main():
    sp = SamplingParams(temperature=0.001, max_tokens=4)

    baseline = LLM(MODEL_PATH, weight_loading_mode="eager")
    out_eager = baseline.generate(PROMPT, sp)

    # three-level, all decoder layers allowed on HBM
    llm3_all = LLM(MODEL_PATH, weight_loading_mode="three_level", num_hbm_layers=None)
    out3_all = llm3_all.generate(PROMPT, sp)
    assert out3_all == out_eager, f"three_level(all) differs: {out3_all!r} vs {out_eager!r}"

    # three-level, only 2 decoder layers resident -> layers are paged CPU<->HBM
    llm3_2 = LLM(MODEL_PATH, weight_loading_mode="three_level", num_hbm_layers=2)
    out3_2 = llm3_2.generate(PROMPT, sp)
    assert out3_2 == out_eager, f"three_level(2) differs: {out3_2!r} vs {out_eager!r}"
    assert llm3_2.model_runner.weight_loader.num_hbm_layers == 2

    # two-level, control by HBM byte budget instead of layer count
    loader_ref = llm3_all.model_runner.weight_loader
    resident_bytes = sum(p.numel() * p.element_size() for _, p in loader_ref.resident_params)
    layer_bytes = sum(p.numel() * p.element_size() for _, p in loader_ref.layer_groups[0])
    budget = resident_bytes + 2 * layer_bytes

    llm2 = LLM(MODEL_PATH, weight_loading_mode="two_level", hbm_budget_bytes=budget)
    out2 = llm2.generate(PROMPT, sp)
    assert out2 == out_eager, f"two_level(budget) differs: {out2!r} vs {out_eager!r}"
    assert llm2.model_runner.weight_loader.num_hbm_layers == 2

    print("PASS: weight loading")
    print(f"  eager={out_eager!r}")
    print(f"  three_level(all) num_hbm={llm3_all.model_runner.weight_loader.num_hbm_layers}")
    print(f"  three_level(2) num_hbm={llm3_2.model_runner.weight_loader.num_hbm_layers}")
    print(f"  two_level(budget) num_hbm={llm2.model_runner.weight_loader.num_hbm_layers} "
          f"budget={budget} resident={resident_bytes} layer={layer_bytes}")


if __name__ == "__main__":
    main()
