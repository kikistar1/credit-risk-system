"""日志工具"""
import os
import sys
from loguru import logger as _logger


def get_logger(name="app"):
    _logger.remove()
    level = os.environ.get("LOG_LEVEL", "INFO")

    _logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> | <level>{message}</level>",
        level=level,
    )

    _logger.add(
        "logs/{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",
        level=level,
        encoding="utf-8",
    )

    return _logger.bind(module=name)