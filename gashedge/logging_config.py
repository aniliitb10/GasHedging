"""Millisecond-precision logging shared by all notebooks and helper modules."""

import logging
import sys

_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once (idempotent) with ms-precision timestamps."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # third-party chatter is not part of the tutorial narrative
    for noisy in ("matplotlib", "PIL", "traitlets", "nbclient"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger, configuring the root logger on first use."""
    configure_logging(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
