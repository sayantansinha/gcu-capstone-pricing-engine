from __future__ import annotations

import base64

import streamlit as st

from src.services.pipeliine.analytics.visual_tools_service import (
    chart_actual_vs_pred, chart_residuals, chart_residuals_qq
)
from src.ui.common import show_last_training_badge


def _get_truth_pred(res):
    # Support both shapes of last_model:
    # { "y_true":..., "y_pred":... } OR { "base": {"y_true":..., "y_pred":...}, ... }
    if isinstance(res, dict):
        if "y_true" in res and "y_pred" in res:
            return res["y_true"], res["y_pred"]
        base = res.get("base", {})
        return base.get("y_true"), base.get("y_pred")
    return None, None


def _display_chart(img_data_uri: str):
    # robust for older/newer Streamlit: turn data-URI → bytes
    if isinstance(img_data_uri, str) and img_data_uri.startswith("data:image"):
        b64 = img_data_uri.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        st.image(img_bytes, use_column_width=True)
    else:
        st.image(img_data_uri, use_column_width=True)


def render():
    st.header("Visual Tools")

    res = st.session_state.get("last_model")
    if not res:
        st.info("Train a model first in **Analytics Tools → Analytical Tools – Model**.")
        return

    show_last_training_badge()

    y_true, y_pred = _get_truth_pred(res)
    if y_true is None or y_pred is None:
        st.error("Could not find y_true / y_pred in the last model results.")
        return

    # --- chart images generated ---
    p1 = chart_actual_vs_pred(y_true, y_pred)
    p2 = chart_residuals(y_true, y_pred)
    p3 = chart_residuals_qq(y_true, y_pred)

    # --- 2-column layout ---
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        _display_chart(p1)

    with c2:
        _display_chart(p2)

    c3, _ = st.columns(2, gap="medium")
    with c3:
        _display_chart(p3)
