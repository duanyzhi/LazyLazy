"""对外的 LLM 类。

这里只是 LLMEngine 的别名，具体实现见 lazy.engine.llm_engine。
"""

from lazy.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """LLM 引擎的对外入口，用法见 LLMEngine。"""
