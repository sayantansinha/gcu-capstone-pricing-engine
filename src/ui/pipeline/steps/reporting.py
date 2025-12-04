from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Final

import streamlit as st

from src.services.pipeline.analytics.reporting_service import Section, build_html_report, build_pdf_from_html
from src.services.pipeline.analytics.visual_tools_service import (
    chart_actual_vs_pred, chart_residuals, chart_residuals_qq
)
from src.ui.common import show_last_training_badge, load_active_cleaned_feature_master_from_session
from src.utils.data_io_utils import latest_report_for_run, load_report_for_download

REPORT_FILENAME_PREFIX: Final[str] = "report_"
REPORT_FILENAME_EXT: Final[str] = ".pdf"


def _create_report_filename(report_name: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{REPORT_FILENAME_PREFIX}{report_name}_{ts}"


def _latest_generated_report(run_id: str) -> Optional[str]:
    """
    Backend-agnostic wrapper.

    LOCAL:
      returns a filesystem path as string
    S3:
      returns an s3://bucket/key URI
    """
    return latest_report_for_run(run_id)


def _show_download_button_for_report(report_path: Path) -> None:
    """
    Given a Path-like object (as used in render()), load the report bytes via
    data_io_utils and expose a Streamlit download button.

    Works for both LOCAL paths and S3 URIs (which may have been wrapped in Path()).
    """
    ref = str(report_path)

    try:
        data = load_report_for_download(ref)
    except Exception as ex:
        st.error(f"Unable to load report for download: {ex}")
        return

    # Derive filename for the download
    if ref.startswith("s3://") or ref.startswith("s3:/"):
        file_name = ref.rsplit("/", 1)[-1]
    else:
        file_name = report_path.name

    st.download_button(
        "Download PDF",
        data=data,
        file_name=file_name,
        mime="application/pdf",
    )


def render():
    st.header("Reporting")
    run_id = st.session_state.run_id
    # Model artifacts (from last training)
    res = st.session_state.get("last_model")
    if not res:
        st.info("Train a model first in **Analytics Tools → Analytical Tools – Model**.")
        return

    show_last_training_badge()

    # Loading the cleaned feature master for this run
    df, fm_clean = load_active_cleaned_feature_master_from_session()

    # Fetch all relevant data
    # Get y_true / y_pred
    y_true = (res or {}).get("y_true")
    y_pred = (res or {}).get("y_pred")

    # Trained models:
    trained_models = (res or {}).get("trained_models")

    # Performance blocks from what you already save
    perf_payload = {
        "per_model_metrics": (res or {}).get("base", {}).get("per_model_metrics"),
        "ensemble_avg_metrics": (res or {}).get("ensemble_avg", {}).get("metrics"),
        "ensemble_wgt_metrics": (res or {}).get("ensemble_wgt", {}).get("metrics"),
    }

    with st.container(border=True):
        st.markdown("Report Selection Options....")
        include_data = st.checkbox("Include dataset preview (first 20 rows)", value=True, disabled=df is None)
        include_plots = st.checkbox("Include diagnostic plots", value=True)
        include_perf = st.checkbox("Include model metrics & BP test", value=True)
        report_name = st.text_input("Report name (no spaces)", value="pricing_engine_summary")

        if st.button("Generate Report", type="primary"):
            sections = []
            meta = {
                "model": ', '.join(trained_models),
                "rows": int(len(df)) if df is not None else None,
                "feature_master_path": fm_clean,
                "run_dir": st.session_state.get("last_model_run_dir"),
            }

            if include_data and df is not None:
                sections.append(Section("Dataset Preview", df.head(20).to_html(index=False)))

            if include_plots:
                imgs = [
                    ("Actual vs Predicted", chart_actual_vs_pred(y_true, y_pred)),
                    ("Residuals vs Predicted", chart_residuals(y_true, y_pred)),
                    ("Residuals Q–Q", chart_residuals_qq(y_true, y_pred))
                ]
                html = "".join([f"<h3>{t}</h3><img src='{src}'/>" for t, src in imgs])
                sections.append(Section("Diagnostics", html))

            if include_perf:
                perf_html = (
                    f"<h3>Performance</h3><pre>{json.dumps(perf_payload, indent=2)}</pre>"
                    # f"<h3>Breusch–Pagan</h3><pre>{json.dumps(res['bp'], indent=2)}</pre>"
                )
                sections.append(Section("Performance & Diagnostics", perf_html))

            html_report = build_html_report("Predictive Pricing Engine – Report", meta, sections)

            # Save report
            report_filenm = _create_report_filename(report_name)
            out_path = build_pdf_from_html(html_report, report_filenm, run_id)

            st.success(f"Saved report: {out_path}")

        # Display info for existing report generated
        latest_generated_report = _latest_generated_report(run_id)
        if latest_generated_report:
            st.session_state["report_generated"] = True
            st.markdown("---")
            st.success(f"Report available for download: **{latest_generated_report}**")
            _show_download_button_for_report(Path(latest_generated_report))
