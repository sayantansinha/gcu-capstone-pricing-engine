from __future__ import annotations

from typing import Any, Dict, List, Optional
import ast
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.services.reports.reports_service import (
    build_portfolio_overview,
    build_territory_intelligence,
    build_platform_strategy,
    build_bundle_value_report,
    build_executive_summary,
    build_model_analytics,
    generate_model_analytics_report_pdf,
)
from src.utils.log_utils import get_logger
from src.utils.model_io_utils import (
    load_model_registry,
    load_model_json,
    load_stacked_model_for_run,
)
from src.utils.data_io_utils import load_report_for_download
from src.utils.explain_utils import permutation_importance_scores, shap_summary_df
from src.services.pipeline.analytics.visual_tools_service import (
    chart_actual_vs_pred,
    chart_residuals,
    chart_residuals_qq,
)

LOGGER = get_logger("ui_reports")


# ---------------------------------------------------------------------
# Pricing / bundling helpers
# ---------------------------------------------------------------------
def _get_pricing_df_from_session() -> Optional[pd.DataFrame]:
    df = st.session_state.get("compare_predictions_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _flatten_bundle_results(bundle_results: List[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for i, bundle in enumerate(bundle_results, start=1):
        bundle_id = getattr(bundle, "bundle_id", None) or f"bundle_{i}"

        for t in getattr(bundle, "titles", []):
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "strategy_name": getattr(bundle, "strategy_name", None),
                    "bundle_price_raw": getattr(bundle, "bundle_price_raw", None),
                    "bundle_value_score": getattr(bundle, "bundle_price_score", None),
                    "bundle_fit_score": getattr(bundle, "bundle_fit_score", None),
                    "bundle_diversity_score": getattr(bundle, "bundle_diversity_score", None),
                    "bundle_risk_score": getattr(bundle, "bundle_risk_score", None),
                    "title_id": t.get("title_id"),
                    "title_name": t.get("title_name"),
                    "predicted_price": t.get("predicted_price"),
                    "region": t.get("region"),
                    "platform": t.get("platform"),
                    "release_year": t.get("release_year"),
                    "genres": t.get("genres"),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _get_bundling_df_from_session() -> Optional[pd.DataFrame]:
    bundle_results = st.session_state.get("bundle_results")
    if not bundle_results:
        return None

    df = _flatten_bundle_results(bundle_results)
    if df.empty:
        return None
    return df


# ---------------------------------------------------------------------
# Model run + last_model helpers
# ---------------------------------------------------------------------
def _select_model_run() -> Optional[str]:
    """
    Select a model for analytics. The underlying key is still run_id,
    but the user sees the model name in the dropdown.
    """
    registry = load_model_registry()
    if not registry:
        st.info(
            "No entries found in the model registry. "
            "Train and register at least one model to view Model Analytics."
        )
        return None

    # Build mapping: run_id -> model_name (best-effort)
    run_ids: List[str] = []
    label_by_run: Dict[str, str] = {}

    for entry in registry:
        rid = entry.get("run_id")
        if not rid:
            continue
        if rid in label_by_run:
            # already captured a label for this run
            continue

        model_name = (
                entry.get("model_display_name")
                or entry.get("model_name")
                or entry.get("name")
                or entry.get("model_key")
                or rid
        )
        label_by_run[rid] = str(model_name)
        run_ids.append(rid)

    run_ids = sorted(run_ids)
    if not run_ids:
        st.info(
            "Model registry is present but contains no valid run_ids. "
            "Check your training / registration pipeline."
        )
        return None

    default_run_id = st.session_state.get("active_run_id")
    default_index = 0
    if default_run_id in run_ids:
        default_index = run_ids.index(default_run_id)

    def _format_run(rid: str) -> str:
        # What the user sees: model name instead of raw run_id
        return label_by_run.get(rid, rid)

    selected_run_id = st.selectbox(
        "Select model for analytics",
        options=run_ids,
        index=default_index,
        format_func=_format_run,
        key="reports_model_analytics_run_id",
    )

    # Stash the model name in session for downstream use (UI label + PDF)
    st.session_state["reports_model_name_for_selected_run"] = label_by_run.get(selected_run_id)
    return selected_run_id


def _get_model_results_from_session() -> Dict[str, Any]:
    """
    Mirror the data-product project's reporting UI: pull rich results
    from st.session_state['last_model'] produced in Analytical Tools.
    """
    last = st.session_state.get("last_model") or {}
    if not isinstance(last, dict):
        return {}

    per_model = last.get("per_model_metrics")
    if per_model is None:
        base = last.get("base") or {}
        per_model = base.get("per_model_metrics")

    ensemble_avg = None
    if "ensemble_avg" in last:
        ensemble_avg = (last.get("ensemble_avg") or {}).get("metrics")
    if ensemble_avg is None:
        ensemble_avg = last.get("ensemble_avg_metrics")

    ensemble_wgt = None
    if "ensemble_wgt" in last:
        ensemble_wgt = (last.get("ensemble_wgt") or {}).get("metrics")
    if ensemble_wgt is None:
        ensemble_wgt = last.get("ensemble_weighted_metrics")

    return {
        "per_model_metrics": per_model,
        "ensemble_avg_metrics": ensemble_avg,
        "ensemble_wgt_metrics": ensemble_wgt,
        "bp_results": last.get("bp"),
        "y_true": last.get("y_true"),
        "y_pred": last.get("y_pred"),
        "model": last.get("model"),
        "X_valid": last.get("X_valid"),
        "y_valid": last.get("y_valid"),
        "X_sample": last.get("X_sample"),
    }


def _download_button_for_report(ref: str, label: str) -> None:
    if not ref:
        st.error("No report reference returned.")
        return
    try:
        pdf_bytes = load_report_for_download(ref)
    except Exception as ex:  # noqa: BLE001
        st.error(f"Unable to load report for download: {ex}")
        return

    if ref.startswith("s3://"):
        filename = ref.rsplit("/", 1)[-1]
    else:
        filename = Path(ref).name

    st.download_button(label, data=pdf_bytes, file_name=filename, mime="application/pdf")


def _metrics_dict_to_df(metrics: Dict[str, Any]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    rows = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            val = f"{v:.4f}"
        else:
            val = str(v)
        rows.append({"Metric": str(k), "Value": val})
    return pd.DataFrame(rows)


def _coerce_explain_df(obj: Any) -> Optional[pd.DataFrame]:
    """
    Try to turn whatever is stored in explain_params.json into a DataFrame.

    Supports a few common JSON shapes:
      - {"columns": [...], "data": [[...], ...]}
      - list[dict]  (records)
      - dict of column -> list
    """
    if obj is None:
        return None

    try:
        if isinstance(obj, dict):
            cols = obj.get("columns")
            data = obj.get("data")
            if cols is not None and data is not None:
                return pd.DataFrame(data, columns=cols)
            # fall back: try treating dict as column -> list
            return pd.DataFrame(obj)

        if isinstance(obj, list):
            if not obj:
                return pd.DataFrame()
            if isinstance(obj[0], dict):
                return pd.DataFrame(obj)
            return pd.DataFrame(obj)
    except Exception:
        return None

    return None


def _coerce_explain_array(obj: Any):
    """
    Try to turn whatever is stored for y_valid into a 1D array-like.
    """
    if obj is None:
        return None
    try:
        import numpy as np  # local import

        if isinstance(obj, (list, tuple)):
            return np.array(obj)
        if isinstance(obj, dict):
            # If stored as {"values": [...]} or similar, try values()
            if "values" in obj and isinstance(obj["values"], (list, tuple)):
                return np.array(obj["values"])
            return np.array(list(obj.values()))
        return np.asarray(obj)
    except Exception:
        return None


def _load_explain_from_artifacts(run_id: str) -> Dict[str, Any]:
    """
    Load explain_params.json for a run and normalize it into
    X_valid, y_valid, X_sample if present.

    Accepts a few common key variants so it is robust to how the JSON was written.
    """
    try:
        params = load_model_json(run_id, "explain_params.json") or {}
    except Exception:
        params = {}

    if not isinstance(params, dict):
        return {}

    # Be forgiving about key names / casing
    def _pick(params_dict: Dict[str, Any], *candidates: str):
        for name in candidates:
            if name in params_dict:
                return params_dict[name]
        return None

    X_valid_raw = _pick(params, "X_valid", "x_valid", "X_val", "x_val")
    y_valid_raw = _pick(params, "y_valid", "yVal", "y_val", "Y_valid")
    X_sample_raw = _pick(params, "X_sample", "x_sample", "X_smpl", "x_smpl")

    X_valid = _coerce_explain_df(X_valid_raw)
    y_valid = _coerce_explain_array(y_valid_raw)
    X_sample = _coerce_explain_df(X_sample_raw)

    return {
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_sample": X_sample,
    }


def _get_model_results(run_id: str, model_name: str) -> Dict[str, Any]:
    """
    Combine session-based last_model with persisted artifacts for the selected run.

    Leverages build_model_analytics(report) as the single source of truth for
    the saved model filename via report.stacked_meta['model_name'].

    Priority:
      - Use st.session_state['last_model'] when available.
      - Only hit disk/S3 when something is missing (model or explain inputs).
      - Load X_valid, y_valid, X_sample from explain_params.json.
      - Load the stacked model <model_name>.joblib using report.stacked_meta['model_name'].
    """
    base = _get_model_results_from_session() or {}

    # 1) Ensure we have a model: prefer session, otherwise load from artifacts
    if base.get("model") is None and model_name:
        model = load_stacked_model_for_run(run_id, model_name)
        if model is not None:
            LOGGER.info(f"Model loaded joblib: {model}")
            base["model"] = model

    # 2) If X_valid, y_valid, X_sample are ALL present in session, don't hit JSON
    have_all_from_session = all(
        base.get(key) is not None for key in ("X_valid", "y_valid", "X_sample")
    )
    if have_all_from_session:
        LOGGER.info(f"Model Results from session: {base}")
        return base

    # 3) Fill in any missing explain inputs from explain_params.json
    explain = _load_explain_from_artifacts(run_id)

    for key in ("X_valid", "y_valid", "X_sample"):
        if base.get(key) is None and explain.get(key) is not None:
            base[key] = explain[key]

    if explain:
        base.setdefault("explain_params", {}).update(explain)

    LOGGER.info(f"Model Results from saved artifacts: {base}")

    return base


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------
def render_reports() -> None:
    st.title("Reports")

    pricing_df = _get_pricing_df_from_session()
    bundling_df = _get_bundling_df_from_session()

    overview = ti = ps = es = None
    if pricing_df is not None:
        overview = build_portfolio_overview(pricing_df, top_n=20)
        ti = build_territory_intelligence(pricing_df)
        ps = build_platform_strategy(pricing_df)
        es = build_executive_summary(pricing_df)

    bundle_report = None
    if bundling_df is not None:
        bundle_report = build_bundle_value_report(bundling_df)

    if pricing_df is not None:
        st.markdown("### Downloads")

        pricing_csv = pricing_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download full predictions (CSV)",
            data=pricing_csv,
            file_name="pricing_predictions_full.csv",
            mime="text/csv",
            key="download_pricing_full_csv",
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("pricing_predictions_full.csv", pricing_df.to_csv(index=False))

            if overview is not None and not overview.by_title.empty:
                zf.writestr("portfolio_by_title.csv", overview.by_title.to_csv(index=False))
            if overview is not None and overview.by_genre is not None and not overview.by_genre.empty:
                zf.writestr("portfolio_by_genre.csv", overview.by_genre.to_csv(index=False))

            if ti is not None and not ti.by_territory.empty:
                zf.writestr("territory_summary.csv", ti.by_territory.to_csv(index=False))

            if ps is not None and not ps.by_platform.empty:
                zf.writestr("platform_summary.csv", ps.by_platform.to_csv(index=False))

            if bundle_report is not None and not bundle_report.bundles.empty:
                zf.writestr("bundle_summary.csv", bundle_report.bundles.to_csv(index=False))

            if es is not None and not es.top_titles.empty:
                zf.writestr("exec_top_titles.csv", es.top_titles.to_csv(index=False))
            if es is not None and not es.top_territories.empty:
                zf.writestr("exec_top_territories.csv", es.top_territories.to_csv(index=False))
            if es is not None and not es.top_platforms.empty:
                zf.writestr("exec_top_platforms.csv", es.top_platforms.to_csv(index=False))

        zip_buf.seek(0)
        st.download_button(
            label="⬇️ Download all report data (ZIP)",
            data=zip_buf,
            file_name="pricing_reports_all.csv.zip",
            mime="application/zip",
            key="download_all_reports_zip",
        )

    else:
        st.info(
            "Pricing reports (Portfolio, Territory, Platform, Executive) "
            "will be available after you run predictions in the Compare tab."
        )

    st.markdown("---")

    tab_model, tab_portfolio, tab_territory, tab_platform, tab_bundle, tab_exec = st.tabs(
        [
            "Model Analytics",
            "Portfolio Overview",
            "Territory Intelligence",
            "Platform Strategy",
            "Bundle Optimization",
            "Executive Summary",
        ]
    )

    # -----------------------------------------------------------------
    # Model Analytics – BP + PI + SHAP BEFORE graphs
    # -----------------------------------------------------------------
    with tab_model:
        st.subheader("Model Analytics Report")

        run_id = _select_model_run()
        if not run_id:
            return

        report = build_model_analytics(run_id)
        if report.predictions.empty and report.per_model_metrics.empty:
            st.info(
                f"No model analytics artifacts found for run_id='{run_id}'. "
                "Ensure `save_model_artifacts` was called for this run."
            )
            return

        # Use model name for display; fall back to run_id if not available
        model_name = st.session_state.get("reports_model_name_for_selected_run") or run_id
        st.markdown(f"#### Selected model: `{model_name}`")

        # High-level diagnostics
        if report.derived_metrics:
            cols = st.columns(4)
            cols[0].metric("RMSE", f"{report.derived_metrics.get('rmse', 0):.3f}")
            cols[1].metric("MAE", f"{report.derived_metrics.get('mae', 0):.3f}")
            r2 = report.derived_metrics.get("r2")
            cols[2].metric("R²", f"{r2:.3f}" if r2 is not None else "n/a")
            mape = report.derived_metrics.get("mape")
            cols[3].metric("MAPE (%)", f"{mape:.2f}" if mape is not None else "n/a")
        else:
            st.info("No predictions.csv found or it is empty; cannot compute RMSE/MAE/R²/MAPE.")

        # Ensemble metrics as tables (keep original header text)
        st.markdown("### Ensemble Metrics (from saved artifacts)")
        if report.ensemble_avg_metrics or report.ensemble_weighted_metrics or report.stacked_meta:
            col_avg, col_wgt, col_stack = st.columns(3)

            with col_avg:
                st.caption("Average Ensemble")
                df_avg = _metrics_dict_to_df(report.ensemble_avg_metrics)
                if not df_avg.empty:
                    st.dataframe(df_avg, hide_index=True, use_container_width=True)
                else:
                    st.write("No metrics found.")

            with col_wgt:
                st.caption("Weighted Ensemble (inv RMSE)")
                df_wgt = _metrics_dict_to_df(report.ensemble_weighted_metrics)
                if not df_wgt.empty:
                    st.dataframe(df_wgt, hide_index=True, use_container_width=True)
                else:
                    st.write("No metrics found.")

            with col_stack:
                st.caption("Stacked Ensemble (hybrid)")
                stack_metrics = (report.stacked_meta or {}).get("metrics", {})
                df_stack = _metrics_dict_to_df(stack_metrics)
                if not df_stack.empty:
                    st.dataframe(df_stack, hide_index=True, use_container_width=True)
                else:
                    st.write("No metrics found.")
        else:
            st.info("No ensemble JSON artifacts found for this run.")

        # Per-model metrics table (strip BP JSON column if present)
        st.markdown("### Per-model Metrics")

        bp_from_per_model: Optional[Dict[str, Any]] = None
        if not report.per_model_metrics.empty:
            per_model_df = report.per_model_metrics.copy()

            last_col = per_model_df.columns[-1]
            series = per_model_df[last_col]

            # Try to parse first non-null cell as dict (JSON or Python repr)
            for cell in series.dropna().tolist():
                if isinstance(cell, dict):
                    bp_from_per_model = cell
                    break

                text = str(cell).strip()
                if not text:
                    continue
                # Try JSON
                if text.startswith("{") and text.endswith("}"):
                    try:
                        bp_from_per_model = json.loads(text)
                        break
                    except Exception:  # noqa: BLE001
                        # Try Python literal dict
                        try:
                            parsed = ast.literal_eval(text)
                            if isinstance(parsed, dict):
                                bp_from_per_model = parsed
                                break
                        except Exception:  # noqa: BLE001
                            continue

            # If we parsed BP successfully, drop the last column from the view
            if isinstance(bp_from_per_model, dict):
                per_model_df = per_model_df.drop(columns=[last_col])

            st.dataframe(per_model_df, use_container_width=True)
        else:
            st.info("No per_model_metrics.csv found for this run.")

        # BP + Permutation Importance + SHAP (before graphs)
        model_results = _get_model_results(run_id, model_name)
        if not model_results and bp_from_per_model is None:
            LOGGER.warning("No model results found for this run.")
            st.info(
                "No detailed model results found. "
                "Train a model in the pipeline Analytical Tools step to populate SHAP/BP."
            )
        else:
            # BP table (prefer per_model JSON; fallback to last_model)
            st.markdown("### Breusch–Pagan test (heteroscedasticity)")
            bp = (
                    bp_from_per_model
                    or (model_results.get("bp_results") if model_results else None)
                    or (model_results.get("bp") if model_results else None)
            )
            if isinstance(bp, dict) and bp:
                bp_df = pd.DataFrame(
                    {"Statistic": list(bp.keys()), "Value": list(bp.values())}
                )
                st.dataframe(bp_df, hide_index=True, use_container_width=True)
            else:
                st.info("Breusch–Pagan results not available for this run.")

            # Permutation importance
            st.markdown("### Permutation Importance (Validation Set)")
            model = model_results.get("model") if model_results else None
            X_valid = model_results.get("X_valid") if model_results else None
            y_valid = model_results.get("y_valid") if model_results else None
            pi_df = None
            if model is not None and X_valid is not None and y_valid is not None:
                LOGGER.info("Calculating Permutation Importance scores")
                try:
                    pi_df = permutation_importance_scores(model, X_valid, y_valid, n_repeats=5)
                    LOGGER.info(f"Calculatied Permutation Importance score DF : {pi_df.shape}")
                except Exception as ex:  # noqa: BLE001
                    st.warning(f"Unable to compute permutation importance: {ex}")
            if pi_df is not None and not pi_df.empty:
                st.dataframe(pi_df.head(25), use_container_width=True)
            else:
                st.info("Permutation Importance metrics not available.")

            # SHAP summary
            st.markdown("### SHAP summary (Mean |SHAP|)")
            X_sample = model_results.get("X_sample") if model_results else None
            LOGGER.info(f"Calculating SHAP scores with sample X : {X_sample.shape}")
            shap_df = None
            if model is not None and X_sample is not None:
                try:
                    shap_df = shap_summary_df(model, X_sample)
                    LOGGER.info(f"Calculated SHAP scores : {shap_df.shape}")
                except Exception as ex:  # noqa: BLE001
                    st.warning(f"Unable to compute SHAP summary: {ex}")
            if shap_df is not None and not shap_df.empty:
                st.dataframe(shap_df.head(25), use_container_width=True)
            else:
                st.info("SHAP summary not available.")

        # Prediction diagnostics & charts (after BP/PI/SHAP)
        st.markdown("### Prediction Diagnostics (y_true vs y_pred)")

        # Tabular + line chart
        if not report.predictions.empty and {"y_true", "y_pred"}.issubset(
                report.predictions.columns
        ):
            pred_df = report.predictions.copy().dropna(subset=["y_true", "y_pred"])
            if not pred_df.empty:
                st.dataframe(pred_df.head(200), use_container_width=True)
                chart_df = pred_df[["y_true", "y_pred"]]
                st.line_chart(chart_df)
            else:
                st.info("predictions.csv exists but has no valid y_true / y_pred rows.")
        else:
            st.info("No predictions.csv found or it does not contain y_true/y_pred.")

        # Residual diagnostics charts from visual_tools_service
        if not report.predictions.empty and {"y_true", "y_pred"}.issubset(
                report.predictions.columns
        ):
            y_true = pd.to_numeric(report.predictions["y_true"], errors="coerce")
            y_pred = pd.to_numeric(report.predictions["y_pred"], errors="coerce")
            mask = y_true.notna() & y_pred.notna()
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if not y_true.empty:
                st.markdown("#### Residual diagnostics")

                uri1 = chart_actual_vs_pred(y_true, y_pred)
                uri2 = chart_residuals(y_true, y_pred)
                uri3 = chart_residuals_qq(y_true, y_pred)

                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Actual vs Predicted")
                    st.image(uri1, use_column_width=True)
                with col2:
                    st.caption("Residuals vs Predicted")
                    st.image(uri2, use_column_width=True)

                # Q–Q plot sized the same as the other residual charts
                col3, col4 = st.columns(2)
                with col3:
                    st.caption("Residuals Q–Q plot")
                    st.image(uri3, use_column_width=True)
                # col4 intentionally left empty to maintain grid alignment
                with col4:
                    st.write("")

        # PDF export
        st.markdown("---")
        st.markdown("#### Export Model Analytics report (PDF)")

        if st.button("Generate Model Analytics PDF", key="btn_model_analytics_pdf"):
            try:
                ref = generate_model_analytics_report_pdf(
                    run_id,
                    report,
                    model_results=model_results if model_results else None,
                    model_name=model_name,
                )
                st.success(f"Model Analytics report generated: {ref}")
                _download_button_for_report(ref, "Download Model Analytics report (PDF)")
            except Exception as ex:  # noqa: BLE001
                st.error(f"Unable to generate Model Analytics PDF: {ex}")

    # -----------------------------------------------------------------
    # Portfolio Pricing Overview
    # -----------------------------------------------------------------
    with tab_portfolio:
        st.subheader("Portfolio Pricing Overview")

        if pricing_df is None or overview is None:
            st.info(
                "Portfolio Overview requires pricing predictions from the Compare tab. "
                "Run a comparison to populate this report."
            )
        else:
            cols = st.columns(4)
            cols[0].metric("Titles (unique)", overview.overall_summary.get("num_titles", 0))
            cols[1].metric(
                "Total Predicted Value",
                f"{overview.overall_summary.get('total_predicted_value', 0):,.0f}",
            )
            cols[2].metric(
                "Average Predicted Price",
                f"{overview.overall_summary.get('avg_predicted_price', 0):,.0f}",
            )
            cols[3].metric(
                "Median Predicted Price",
                f"{overview.overall_summary.get('median_predicted_price', 0):,.0f}",
            )

            st.markdown("Top Titles by Predicted Value")
            if not overview.by_title.empty:
                st.dataframe(overview.by_title, use_container_width=True)

                portfolio_titles_csv = overview.by_title.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download portfolio by title (CSV)",
                    data=portfolio_titles_csv,
                    file_name="portfolio_by_title.csv",
                    mime="text/csv",
                    key="download_portfolio_by_title_csv",
                )
            else:
                st.info(
                    "No title-level aggregation available. "
                    "Make sure your Compare predictions include title_id or title_name."
                )

    # -----------------------------------------------------------------
    # Territory Intelligence
    # -----------------------------------------------------------------
    with tab_territory:
        st.subheader("Territory Intelligence Report")

        if pricing_df is None or ti is None:
            st.info(
                "Territory Intelligence requires pricing predictions with a territory or "
                "region column. Run the Compare tab with these fields to enable this view."
            )
        else:
            if ti.by_territory.empty:
                st.info(
                    "This report needs either a territory or region column in the "
                    "Compare predictions. Currently neither was found."
                )
            else:
                st.markdown("Territory Summary")
                st.dataframe(ti.by_territory, use_container_width=True)

                territory_csv = ti.by_territory.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download territory summary (CSV)",
                    data=territory_csv,
                    file_name="territory_summary.csv",
                    mime="text/csv",
                    key="download_territory_summary_csv",
                )

                if "avg_predicted" in ti.by_territory.columns:
                    st.markdown("Average Predicted Price by Territory")
                    chart_df = ti.by_territory.set_index("territory")["avg_predicted"]
                    st.bar_chart(chart_df)

    # -----------------------------------------------------------------
    # Platform Strategy
    # -----------------------------------------------------------------
    with tab_platform:
        st.subheader("Platform Strategy Report")

        if pricing_df is None or ps is None:
            st.info(
                "Platform Strategy requires pricing predictions with a platform column. "
                "Run the Compare tab including platform information to enable this view."
            )
        else:
            if ps.by_platform.empty:
                st.info(
                    "This report needs a platform column in the Compare predictions. "
                    "Currently no platform information was found."
                )
            else:
                st.markdown("Platform Summary")
                st.dataframe(ps.by_platform, use_container_width=True)

                platform_csv = ps.by_platform.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download platform summary (CSV)",
                    data=platform_csv,
                    file_name="platform_summary.csv",
                    mime="text/csv",
                    key="download_platform_summary_csv",
                )

                if "avg_predicted" in ps.by_platform.columns:
                    st.markdown("Average Predicted Price by Platform")
                    chart_df = ps.by_platform.set_index("platform")["avg_predicted"]
                    st.bar_chart(chart_df)

    # -----------------------------------------------------------------
    # Bundle Value Optimization
    # -----------------------------------------------------------------
    with tab_bundle:
        st.subheader("Bundle Value Optimization Report")

        if bundling_df is None:
            st.info(
                "No bundling results found in this session. "
                "Run the Bundling tab in Compare & Bundling to populate this report."
            )
        else:
            if bundle_report is None or bundle_report.bundles.empty:
                st.info(
                    "Bundling results are present but do not contain bundle_id and "
                    "bundle_price_raw. Update the bundling output schema or adjust "
                    "build_bundle_value_report if your schema differs."
                )
            else:
                cols = st.columns(3)
                cols[0].metric("Bundles", bundle_report.summary.get("num_bundles", 0))
                cols[1].metric(
                    "Total Bundle Value",
                    f"{bundle_report.summary.get('total_bundle_value', 0):,.0f}",
                )
                cols[2].metric(
                    "Average Bundle Value",
                    f"{bundle_report.summary.get('avg_bundle_value', 0):,.0f}",
                )

                st.markdown("Bundle-Level Summary")
                st.dataframe(bundle_report.bundles, use_container_width=True)

                bundles_csv = bundle_report.bundles.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download bundle summary (CSV)",
                    data=bundles_csv,
                    file_name="bundle_summary.csv",
                    mime="text/csv",
                    key="download_bundle_summary_csv",
                )

    # -----------------------------------------------------------------
    # Executive Pricing Intelligence Summary
    # -----------------------------------------------------------------
    with tab_exec:
        st.subheader("Executive Pricing Intelligence Summary")

        if pricing_df is None or es is None:
            st.info(
                "Executive Summary requires pricing predictions from the Compare tab. "
                "Run a comparison to populate this report."
            )
        else:
            st.markdown("Top Titles by Predicted Value")
            if not es.top_titles.empty:
                st.dataframe(es.top_titles, use_container_width=True)

                exec_titles_csv = es.top_titles.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download top titles (CSV)",
                    data=exec_titles_csv,
                    file_name="exec_top_titles.csv",
                    mime="text/csv",
                    key="download_exec_top_titles_csv",
                )
            else:
                st.info(
                    "No title-level information found. "
                    "Ensure title_id/title_name and predicted_price are present in Compare predictions."
                )

            st.markdown("Top Territories by Total Predicted Value")
            if not es.top_territories.empty:
                st.dataframe(es.top_territories, use_container_width=True)

                exec_territories_csv = es.top_territories.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download top territories (CSV)",
                    data=exec_territories_csv,
                    file_name="exec_top_territories.csv",
                    mime="text/csv",
                    key="download_exec_top_territories_csv",
                )
            else:
                st.info(
                    "No territory-level information found. "
                    "Add territory or region to the Compare predictions to enable this view."
                )

            st.markdown("Top Platforms by Total Predicted Value")
            if not es.top_platforms.empty:
                st.dataframe(es.top_platforms, use_container_width=True)

                exec_platforms_csv = es.top_platforms.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download top platforms (CSV)",
                    data=exec_platforms_csv,
                    file_name="exec_top_platforms.csv",
                    mime="text/csv",
                    key="download_exec_top_platforms_csv",
                )
            else:
                st.info(
                    "No platform-level information found. "
                    "Add platform to the Compare predictions to enable this view."
                )
