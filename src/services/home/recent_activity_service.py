from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

_ACTIVITY_KEY = "recent_activity"


def _ensure_activity_list() -> List[Dict[str, Any]]:
    """
    Ensure we have a list for recent activity in session_state and return it.
    Stored per Streamlit session; capped by add_recent_activity.
    """
    if _ACTIVITY_KEY not in st.session_state:
        st.session_state[_ACTIVITY_KEY] = []
    return st.session_state[_ACTIVITY_KEY]


def add_recent_activity(
        label: str,
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a recent-activity entry to the head of the list.

    Example usage from other pages:
        add_recent_activity("Viewed Price Predictor")
        add_recent_activity("Ran pipeline", detail=run_id)
    """
    items = _ensure_activity_list()
    entry = {
        "label": label,
        "detail": detail,
        "metadata": metadata or {},
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    # newest first
    items.insert(0, entry)
    # cap to last 20 events for this session
    del items[20:]


def get_recent_activity(max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Return at most `max_items` recent activity entries (newest first).
    """
    items = _ensure_activity_list()
    return items[:max_items]
