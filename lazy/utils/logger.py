"""统一的日志配置。

每个模块用 init_logger(__name__) 拿 logger，都挂到 "lazy" 根 logger 下，
级别由环境变量 LAZY_LOG_LEVEL 控制。
"""

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_handler: logging.Handler | None = None
_level: str | int = logging.INFO


def _get_handler() -> logging.Handler:
    """拿到全局唯一的 stream handler，第一次调用时创建。"""
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
    """按模块名拿一个挂好 handler 的 logger。

    包外调用方（比如测试脚本的 __main__）不在 "lazy" 命名空间下，这里直接
    给它挂上 handler，免得日志丢失。

    Args:
        name: 模块名，通常传 __name__。

    Returns:
        配置好的 logger。
    """
    handler = _get_handler()
    logger = logging.getLogger(name)
    if not name.startswith("lazy"):
        if handler not in logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(_level)
        logger.propagate = False
    return logger
