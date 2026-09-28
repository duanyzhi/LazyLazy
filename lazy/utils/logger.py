import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_handler = None
_level = logging.INFO


def _get_handler():
    global _handler, _level
    if _handler is not None:
        return _handler
    _level = os.getenv("LAZY_LOG_LEVEL", "INFO").upper()
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root = logging.getLogger("lazy")
    root.setLevel(_level)
    root.addHandler(_handler)
    root.propagate = False
    return _handler


def init_logger(name: str) -> logging.Logger:
    """每个模块用 init_logger(__name__) 拿 logger，统一挂到 "lazy" 根 logger 下。

    日志级别用环境变量 LAZY_LOG_LEVEL 控制，默认 INFO。
    """
    handler = _get_handler()
    logger = logging.getLogger(name)
    if not name.startswith("lazy"):
        # 包外调用方（如测试脚本的 __main__）不在 "lazy" 命名空间下，直接挂 handler
        if handler not in logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(_level)
        logger.propagate = False
    return logger
