"""Единая настройка логирования через loguru.

Вся кодовая база пишет через `from loguru import logger`. Stdlib-логи
(uvicorn / fastapi / taskiq / pymilvus / neo4j / httpx / aio_pika)
перехватываются ``InterceptHandler`` и маршрутизируются в тот же loguru
sink — одна система форматирования и JSON-вывода.

Вызвать `setup_logging()` один раз при старте процесса (FastAPI lifespan
и taskiq worker startup — оба импортируют эту функцию).
"""

from __future__ import annotations

import logging
import sys

from loguru import logger

from src.config import settings


class InterceptHandler(logging.Handler):
    """Пробрасывает любые записи stdlib-logging в loguru.

    Стандартный сниппет из loguru-docs
    (https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging).
    """

    def emit(self, record: logging.LogRecord) -> None:
        # уровень loguru по числовому levelno
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # откатываемся к реальному источнику вызова, а не к самому хэндлеру
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )


_STDLIB_LOGGERS_TO_INTERCEPT = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "taskiq",
    "taskiq.worker",
    "taskiq.receiver.receiver",
    "taskiq.retry_middleware",
    "taskiq.process-manager",
    "taskiq.dependencies.ctx",
    "pymilvus",
    "neo4j",
    "httpx",
    "httpcore",
    "asyncio",
    "aio_pika",
    "aiormq",
    "nano-vectordb",
)


def setup_logging() -> None:
    """Configure loguru as the single logging sink for the process.

    Safe to call multiple times — `logger.remove()` + fresh handlers.
    """
    logger.remove()
    level = settings.api.log_level.upper()

    if settings.api.env == "development":
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
                "<dim>{extra}</dim> | {message}"
            ),
            colorize=True,
            backtrace=True,
            diagnose=False,  # True выдаёт локальные значения — небезопасно в shared logs
        )
    else:
        # prod: одна JSON-строка на запись, пригодна для сбора в ELK / Loki
        logger.add(sys.stderr, level=level, serialize=True)

    # перехват stdlib-логов
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in _STDLIB_LOGGERS_TO_INTERCEPT:
        lg = logging.getLogger(name)
        lg.handlers = [InterceptHandler()]
        lg.propagate = False

    logger.info(
        "logging configured  env={env}  level={level}  json={json}",
        env=settings.api.env,
        level=level,
        json=settings.api.env != "development",
    )
