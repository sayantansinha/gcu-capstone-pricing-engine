import logging
import sys
import traceback
from functools import wraps

import streamlit as st


def get_logger(name: str, logging_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)s | file=%(name)s | line_no=%(lineno)d | msg=%(message)s",
            datefmt="%d-%b-%Y %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(logging_level)
    return logger


def handle_streamlit_exception(exc_type, exc_value, exc_traceback):
    """Logs all uncaught exceptions without killing the Streamlit app."""
    logger = get_logger("streamlit-exception")
    if issubclass(exc_type, KeyboardInterrupt):
        # allow Ctrl+C without traceback noise
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(f"Uncaught exception:\n{tb}")

    # Optionally show sanitized message in the Streamlit UI
    st.error(f"An unexpected error occurred: {exc_value}")
    st.info("See logs for full details.")


def streamlit_safe(fn):
    """Decorator to wrap Streamlit page renderers safely with automatic logging."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        logger = get_logger(fn.__module__)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in {fn.__name__}:\n{tb}")
            import streamlit as st
            st.error(f"Error in {fn.__name__}: {e}")
            st.info("Details written to log file.")

    return wrapper
