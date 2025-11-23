from typing import Dict, Final

import streamlit as st

from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_pipeline_flow")

STATUS_CONFIG: Final[Dict[str, Dict[str, str]]] = {
    "not_started": {
        "label": "Not started",
        "bg_color": "#E0E0E0"
    },
    "in_progress": {
        "label": "In progress",
        "bg_color": "#FFE082"
    },
    "error": {
        "label": "Error",
        "bg_color": "#FFCDD2"
    },
    "done": {
        "label": "Done",
        "bg_color": "#C8E6C9"
    }
}


def _flow_status_from_context(ctx: Dict) -> Dict[str, str]:
    """
    Derive per-step status from your existing state/artifacts.
    Allowed statuses: 'not_started', 'in_progress', 'done', 'error'
    """
    return {
        "data_staging": "done" if ctx.get("files_staged") else "not_started",
        "feature_master": "done" if ctx.get("feature_master_exists") else "not_started",
        "display_data": "done" if ctx.get("data_displayed") else "not_started",
        "exploration": "done" if ctx.get("eda_performed") else "not_started",
        "preprocessing": "done" if ctx.get("preprocessing_performed") else "not_started",
        "modeling": "done" if ctx.get("model_trained") else "not_started",
        "visual_tools": "done" if ctx.get("model_trained") else "not_started",
        "reporting": "done" if ctx.get("report_generated") else "not_started"
    }


def _flow_graphviz_dot(status_map: Dict[str, str]) -> str:
    """
    Build a left-to-right graph that mirrors your UI steps.
    Clicking a node sets session_state['jump_to'] = node_id (handled below).
    """
    label_map = {
        "data_staging": "Data Staging",
        "feature_master": "Feature Master",
        "display_data": "Display Data",
        "exploration": "EDA",
        "preprocessing": "Preprocessing",
        "modeling": "Modeling",
        "visual_tools": "Visual Tools",
        "reporting": "Reporting"
    }

    steps = list(label_map.keys())

    # Build nodes
    nodes = []
    for s in steps:
        fill = STATUS_CONFIG.get(status_map.get(s, "not_started")).get("bg_color", "#E0E0E0")
        label = label_map.get(s, s)
        # Use HTML-like label to add a subtle title break
        nodes.append(f'"{s}" [shape=rect, style="filled,rounded", fillcolor="{fill}", label=<{label}>]')

    # Build edges
    edges = []
    for a, b in zip(steps[:-1], steps[1:]):
        edges.append(f'"{a}" -> "{b}"')

    dot = [
        'digraph pipeline {',
        'rankdir=LR;',
        'node [fontname="Inter", fontsize=12, margin="0.15,0.08"];',
        'edge [color="#B0BEC5"];',
        *nodes,
        *edges,
        '}'
    ]
    return "\n".join(dot)


def render_pipeline_flow(ctx: Dict):
    LOGGER.info(f"Rendering pipeline flow with context => {ctx}")
    status_map = _flow_status_from_context(ctx)
    dot = _flow_graphviz_dot(status_map)
    st.graphviz_chart(dot, use_container_width=True)

    # Legend
    # c1, c2, c3, c4, _ = st.columns([0.4, 0.4, 0.4, 0.4, 4])
    c1, c2, _ = st.columns([0.4, 0.4, 4])

    legend_style = (
        "font-size: 12px;"
        "padding: 3px 4px;"
        "border-radius: 4px;"
        "text-align: center;"
        "margin-bottom: 10px;"
    )

    with c1:
        st.markdown(
            f'<div style="background: {STATUS_CONFIG.get("not_started").get("bg_color")};{legend_style}">'
            f'{STATUS_CONFIG.get("not_started").get("label")}</div>',
            unsafe_allow_html=True,
        )
    # with c2:
    #     st.markdown(
    #         f'<div style="background: {STATUS_CONFIG.get("in_progress").get("bg_color")};{legend_style}">'
    #         f'{STATUS_CONFIG.get("in_progress").get("label")}</div>',
    #         unsafe_allow_html=True,
    #     )
    # with c3:
    #     st.markdown(
    #         f'<div style="background: {STATUS_CONFIG.get("error").get("bg_color")};{legend_style}">'
    #         f'{STATUS_CONFIG.get("error").get("label")}</div>',
    #         unsafe_allow_html=True,
    #     )
    with c2:
        st.markdown(
            f'<div style="background: {STATUS_CONFIG.get("done").get("bg_color")};{legend_style}">'
            f'{STATUS_CONFIG.get("done").get("label")}</div>',
            unsafe_allow_html=True,
        )
