"""Console + file logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: str | Path | None = None, level: int | str = logging.INFO) -> logging.Logger:
    """Configure the root logger with a console handler and an optional file handler.

    Calling this repeatedly replaces previous handlers (no duplicated lines).

    Returns:
        The ``deinteract`` logger to be used by scripts.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Keep third-party libraries quiet.
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("deinteract")
