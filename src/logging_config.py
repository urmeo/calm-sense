import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog

from .config import LOG_FILE, LOG_LEVEL

_configured_settings: Optional[tuple[str, Optional[Path], bool]] = None

_shared_processors: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def setup_logging(
    level: str = LOG_LEVEL, log_file: Optional[Path] = LOG_FILE, console: bool = True
) -> None:
    global _configured_settings

    level = level.upper()
    log_level = logging.getLevelNamesMapping().get(level)
    if log_level is None:
        raise ValueError(f"Unknown logging level: {level}")
    log_file = Path(log_file).resolve() if log_file is not None else None
    settings = (level, log_file, console)
    if settings == _configured_settings:
        return

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=_shared_processors,
    )

    handlers: list[logging.Handler] = []
    try:
        if console:
            handlers.append(logging.StreamHandler(sys.stdout))
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        for handler in handlers:
            handler.setFormatter(formatter)
    except Exception:
        for handler in handlers:
            handler.close()
        raise

    project_logger = logging.getLogger("calmsense")
    for existing in project_logger.handlers[:]:
        project_logger.removeHandler(existing)
        existing.close()
    project_logger.setLevel(log_level)
    project_logger.propagate = False
    for handler in handlers:
        project_logger.addHandler(handler)

    _configured_settings = settings


def get_logger(name: str):
    if _configured_settings is None:
        setup_logging()

    if name.startswith("src."):
        name = "calmsense." + name[4:]
    elif name != "calmsense" and not name.startswith("calmsense."):
        name = f"calmsense.{name}"

    return structlog.wrap_logger(
        logging.getLogger(name),
        processors=_shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class LoggerMixin:
    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger
