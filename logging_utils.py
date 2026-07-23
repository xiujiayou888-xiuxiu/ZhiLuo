# -*- coding: utf-8 -*-
"""Shared rotating logger for ZhiLuo modules."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str = "zhiluo") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level_name = os.environ.get("ZHILUO_LOG_LEVEL", "INFO").strip().upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    try:
        from kb_config import get_data_dir
        log_dir = Path(get_data_dir()) / "logs"
    except Exception:
        log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / ("%s.log" % name.replace(".", "_")),
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return logger
