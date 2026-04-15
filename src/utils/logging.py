import logging
import sys

from src.config import settings


def setup_logging():
    logging.basicConfig(
        level=settings.api.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
