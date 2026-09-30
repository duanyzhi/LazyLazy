"""单跑 + 拼 batch 的回归测试。

用法: cd tests && python llm_test.py
"""

from lazy import LLM
from lazy import SamplingParams
from lazy.utils.logger import init_logger

logger = init_logger(__name__)

MODEL_PATH = "/mnt/mizar/models/Qwen3-0.6B-Base"
PROMPTS = [
    "what is the largest animal in the world?",
    "what is the capital of France?",
]

# greedy 期望输出固化为基线（已和 HF generate(do_sample=False) 对齐）。
# 单跑基线：B=1 时 padding mask 全 False，是精确 no-op 路径。
# France 首 token 是 ' The' 而不是旧基线的 '?\n'：两者 logit 打平（bf16 都是
# 18.5），纯 argmax 按 token id 取 ' The'，旧的带种子采样是掷骰子掷出来的。
# 基线太长没法折行，属于规范里允许的模块级长字符串常量。
EXPECTED_SINGLE = [
    "?\nThe largest animal in the world is the blue whale, which can grow up to 100 feet (30 meters) in length and weigh up to 200 tons (180 metric tons). However, the largest animal in the world by weight is the blue whale, which can weigh up",  # noqa: E501
    " The capital of France is **Paris**. It is the largest city in France and serves as the political, cultural, and economic center of the country.<|endoftext|>",  # noqa: E501
]

# 拼 batch 基线：bf16 下 batched matmul 归约顺序不同，在近同分点翻牌是预期
# 行为。目前只有 animal 这条在 "by weight"/"in terms of body mass" 处翻了
# 一次，其余和单跑逐 token 一致。
EXPECTED_BATCHED = [
    "?\nThe largest animal in the world is the blue whale, which can grow up to 100 feet (30 meters) in length and weigh up to 200 tons (180 metric tons). However, the largest animal in terms of body mass is the blue whale, which can weigh up",  # noqa: E501
    " The capital of France is **Paris**. It is the largest city in France and serves as the political, cultural, and economic center of the country.<|endoftext|>",  # noqa: E501
]


def main() -> None:
    """跑单跑回归和拼 batch 回归。"""
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(MODEL_PATH)

    # B=1 单跑回归：验证 mask no-op 路径数值不漂移。
    for prompt, expected in zip(PROMPTS, EXPECTED_SINGLE):
        single_output = llm.generate(prompt, sampling_params)[0]
        assert single_output == expected, (
            f"single-run mismatch!\nexpected: {expected!r}\n"
            f"actual:   {single_output!r}")

    # 多请求一起跑：decode 拼 batch + batch 收缩，France 先 hit_eos，
    # animal 继续到 hit_max。
    batch_outputs = llm.generate(PROMPTS, sampling_params)
    logger.info("batch outputs: %s", batch_outputs)

    # 与固化基线逐条对比：验证多请求不串话 + 数值不漂移。
    for expected, actual in zip(EXPECTED_BATCHED, batch_outputs):
        assert actual == expected, (
            f"mismatch!\nexpected: {expected!r}\nactual:   {actual!r}")

    # cache 应按 seq 释放干净。
    assert not llm.model_runner.kv_caches, "kv cache leaked"

    logger.info("all checks passed")


if __name__ == "__main__":
    main()
