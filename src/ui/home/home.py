from __future__ import annotations

import streamlit as st

from src.services.home.home_service import (
    get_latest_run_info,
    get_latest_model_insights,
    get_system_health,
)
from src.services.home.recent_activity_service import get_recent_activity
from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_home")


def _format_metric_value(val) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        return f"{v:.3f}"
    except Exception:  # defensive
        return str(val)


# ---------------------- CARD RENDERERS ---------------------- #


def _render_application_overview_card() -> None:
    card_html = """
    <div class="ppe-card">
      <h3>Pricing &amp; Licensing Insights</h3>
      <p>
        The application helps Media &amp; Entertainment stakeholders estimate and compare licensing
        prices across territories, platforms, and deal types. It orchestrates a repeatable pipeline
        that ingests raw licensing and title metadata, engineers pricing-relevant features, trains
        and evaluates multiple models, and surfaces deal-level pricing insights through the
        Price Predictor and Compare &amp; Bundling views.
      </p>
      <p class="ppe-caption">
        Use the sidebar to navigate to the Model Pipeline, Price Predictor, and Compare &amp; Bundling modules.
      </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def _render_latest_pipeline_run_card() -> None:
    run_info = get_latest_run_info()

    if not run_info:
        card_html = """
        <div class="ppe-card">
          <h3>Latest Pipeline Run</h3>
          <p>No pipeline runs found yet. Run the pipeline from the <strong>Model Pipeline</strong> section.</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        return

    chips = []

    chips.append(
        f'<span class="ppe-chip {"ppe-chip-ok" if run_info.has_raw else "ppe-chip-warn"}">'
        f'Raw: {"&#10003;" if run_info.has_raw else "&middot;"}'
        f"</span>"
    )
    chips.append(
        f'<span class="ppe-chip {"ppe-chip-ok" if run_info.has_feature_master else "ppe-chip-warn"}">'
        f'Feature master: {"&#10003;" if run_info.has_feature_master else "&middot;"}'
        f"</span>"
    )
    chips.append(
        f'<span class="ppe-chip {"ppe-chip-ok" if run_info.has_feature_master_cleaned else "ppe-chip-warn"}">'
        f'Cleaned master: {"&#10003;" if run_info.has_feature_master_cleaned else "&middot;"}'
        f"</span>"
    )
    chips.append(
        f'<span class="ppe-chip {"ppe-chip-ok" if run_info.has_model else "ppe-chip-warn"}">'
        f'Model: {"&#10003;" if run_info.has_model else "&middot;"}'
        f"</span>"
    )
    chips.append(
        f'<span class="ppe-chip {"ppe-chip-ok" if run_info.has_registered_model else "ppe-chip-warn"}">'
        f'Registered: {"&#10003;" if run_info.has_registered_model else "&middot;"}'
        f"</span>"
    )

    chips_html = " ".join(chips)

    card_html = f"""
    <div class="ppe-card">
      <h3>Latest Pipeline Run</h3>
      <p><strong>Run ID:</strong> <code>{run_info.run_id}</code></p>
      <div class="ppe-chip-row">
        {chips_html}
      </div>
      <p class="ppe-caption">
        Stage indicators are derived from raw/processed storage, model artifacts, and the global model registry.
      </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def _render_most_recent_model_insights_card() -> None:
    insights = get_latest_model_insights()

    if not insights:
        card_html = """
        <div class="ppe-card">
          <h3>Most Recent Model Insights</h3>
          <p>No model artifacts found yet. Complete a modeling run in the <strong>Model Pipeline</strong>.</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        return

    rmse = _format_metric_value(insights.get("rmse"))
    mae = _format_metric_value(insights.get("mae"))
    r2 = _format_metric_value(insights.get("r2"))

    bp = insights.get("bp_summary")
    bp_text = "Breusch–Pagan test summary not available in stored metrics for this run."
    if isinstance(bp, dict):
        stat = bp.get("stat")
        pval = bp.get("pvalue")
        try:
            pval_f = float(pval) if pval is not None else None
        except Exception:
            pval_f = None

        if pval_f is not None:
            if pval_f < 0.05:
                bp_text = (
                    f"Breusch–Pagan test suggests <strong>possible heteroscedasticity</strong> "
                    f"(stat={stat}, p={pval_f:.3g})."
                )
            else:
                bp_text = (
                    f"Breusch–Pagan test does <strong>not show strong evidence of heteroscedasticity</strong> "
                    f"(stat={stat}, p={pval_f:.3g})."
                )
        else:
            bp_text = f"Breusch–Pagan test summary available: stat={stat}, p={pval}."

    card_html = f"""
    <div class="ppe-card">
      <h3>Most Recent Model Insights</h3>
      <p><strong>Run ID:</strong> <code>{insights["run_id"]}</code></p>
      <p><strong>Model:</strong> <code>{insights["model_name"]}</code></p>

      <div class="ppe-metric-row">
        <div class="ppe-metric">
          <div class="ppe-metric-label">RMSE</div>
          <div class="ppe-metric-value">{rmse}</div>
        </div>
        <div class="ppe-metric">
          <div class="ppe-metric-label">MAE</div>
          <div class="ppe-metric-value">{mae}</div>
        </div>
        <div class="ppe-metric">
          <div class="ppe-metric-label">R²</div>
          <div class="ppe-metric-value">{r2}</div>
        </div>
      </div>

      <p class="ppe-caption">{bp_text}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def _render_system_health_card() -> None:
    health = get_system_health()

    env = health["environment"]
    io_backend = health["io_backend"]
    config_ok = health["config_ok"]
    missing = health["missing_config"]
    s3_enabled = health["s3_enabled"]
    s3_error = health["s3_error"]
    app_version = health["app_version"]

    cfg_class = "ppe-chip-ok" if config_ok else "ppe-chip-bad"
    cfg_text = "Config: OK" if config_ok else "Config: missing keys"

    if io_backend.upper() == "S3":
        if s3_enabled and not s3_error:
            s3_chip = '<span class="ppe-chip ppe-chip-ok">S3 connectivity: OK</span>'
        else:
            s3_chip = '<span class="ppe-chip ppe-chip-bad">S3 connectivity: ERROR</span>'
    else:
        s3_chip = '<span class="ppe-chip ppe-chip-warn">S3 connectivity: N/A (LOCAL)</span>'

    missing_text = ""
    if not config_ok and missing:
        missing_text = (
            f"<p class='ppe-caption'>Missing configuration values: {', '.join(missing)}</p>"
        )

    s3_error_text = ""
    if s3_error:
        s3_error_text = f"<p class='ppe-caption'>S3 error: {s3_error}</p>"

    card_html = f"""
    <div class="ppe-card">
      <h3>System Health</h3>
      <p><strong>Environment:</strong> <code>{env}</code></p>
      <p><strong>IO backend:</strong> <code>{io_backend}</code></p>
      <p><strong>App version:</strong> <code>{app_version}</code></p>

      <div class="ppe-chip-row">
        <span class="ppe-chip {cfg_class}">{cfg_text}</span>
        {s3_chip}
      </div>

      {missing_text}
      {s3_error_text}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def _render_recent_activity_card() -> None:
    items = get_recent_activity(max_items=5)

    if not items:
        card_html = """
        <div class="ppe-card">
          <h3>Recent Activity</h3>
          <p class="ppe-caption">No recent activity recorded for this session yet.</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        return

    list_items_html = ""
    for entry in items:
        label = entry.get("label", "")
        detail = entry.get("detail") or ""
        ts = entry.get("ts") or ""
        detail_part = f" – <span class='ppe-activity-detail'>{detail}</span>" if detail else ""
        ts_part = f"<span class='ppe-activity-ts'>{ts}</span>" if ts else ""
        list_items_html += (
            f"<li><span class='ppe-activity-label'>{label}</span>{detail_part}"
            f"<br/>{ts_part}</li>"
        )

    card_html = f"""
    <div class="ppe-card">
      <h3>Recent Activity</h3>
      <ul class="ppe-activity-list">
        {list_items_html}
      </ul>
      <p class="ppe-caption">
        Recent actions are tracked for this browser session only.
      </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------- PUBLIC ENTRYPOINT ---------------------- #


def render_home() -> None:
    """
    Main entrypoint for the Home page.
    CSS for .ppe-* classes is expected to be loaded via src/styles/home.css from app.py.
    """
    st.title("Home")

    # Row 1 – Overview (full width)
    _render_application_overview_card()

    # Row 2 – two cards side-by-side (pipeline + model)
    col1, col2 = st.columns(2)
    with col1:
        _render_latest_pipeline_run_card()
    with col2:
        _render_most_recent_model_insights_card()

    # Row 3 – two cards side-by-side (health + activity)
    col3, col4 = st.columns(2)
    with col3:
        _render_system_health_card()
    with col4:
        _render_recent_activity_card()
