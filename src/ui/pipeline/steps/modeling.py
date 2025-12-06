from __future__ import annotations

import ast
import os
import base64
from copy import deepcopy
from pathlib import Path
from typing import Final, Dict

import pandas as pd
import streamlit as st

from src.services.pipeline.analytics.modeling_service import (
    train_base_models,
    combine_average,
    combine_weighted_inverse_rmse,
    AVAILABLE_MODELS,
    build_stacked_ensemble,
)
from src.services.pipeline.analytics.visual_tools_service import (
    chart_actual_vs_pred,
    chart_residuals,
    chart_residuals_qq,
)
from src.ui.common import show_last_training_badge, store_last_model_info_in_session
from src.ui.common import store_last_run_model_dir_in_session
from src.utils.log_utils import streamlit_safe, get_logger
from src.utils.model_io_utils import (
    save_model_artifacts,
    register_model_in_registry,
    load_model_registry,
)

LOGGER = get_logger("ui_modeling")
PRED_SRC_DISP_NAMES: Final[Dict[str, str]] = {
    "stacked_ensemble": "Stacked Ensemble",
    "weighted_ensemble": "Weighted Ensemble",
    "average_ensemble": "Simple Average Ensemble",
}
METRIC_NAMES: Final[Dict[str, str]] = {
    "RMSE": "Root Mean Squared Error",
    "MAE": "Mean Absolute Error",
    "R2": "R² (Coefficient of Determination)",
}
_BP_TEST_RES_KEYS = {"lm", "lm_pvalue", "f", "f_pvalue"}


def _compare_model_selection_from_last_run(selected_models: list[str]):
    if st.session_state.get("last_model"):
        last = st.session_state.get("last_model")["trained_models"]
        if set(map(str, selected_models)) != set(map(str, last)):
            st.info("Current model selection differs from the last trained set.")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _model_param_ui(algo: str) -> dict:
    """Param UI for allowed models."""
    params: dict = {}
    if algo in ("Ridge", "Lasso"):
        params["alpha"] = st.number_input(
            f"{algo} alpha", 0.0001, 10.0, 1.0, key=f"{algo}_alpha"
        )
    elif algo == "XGBoost":
        params["n_estimators"] = st.slider(
            "XGBoost n_estimators", 100, 1500, 600, 50, key="xgb_ne"
        )
        params["max_depth"] = st.slider("XGBoost max_depth", 2, 12, 6, key="xgb_md")
        params["learning_rate"] = st.number_input(
            "XGBoost learning_rate", 0.01, 0.5, 0.05, key="xgb_lr"
        )
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.9
        params["random_state"] = 42
        params["tree_method"] = "hist"
    elif algo == "LightGBM":
        params["n_estimators"] = st.slider(
            "LightGBM n_estimators", 100, 2000, 800, 50, key="lgb_ne"
        )
        params["learning_rate"] = st.number_input(
            "LightGBM learning_rate", 0.005, 0.5, 0.05, key="lgb_lr"
        )
        params["max_depth"] = st.number_input(
            "LightGBM max_depth (-1=none)", -1, 64, -1, 1, key="lgb_md"
        )
        params["random_state"] = 42
    elif algo == "MLP":
        hl = st.text_input(
            "MLP hidden_layer_sizes (e.g., 256,128)", value="256,128", key="mlp_hl"
        )
        params["hidden_layer_sizes"] = tuple(
            int(x.strip()) for x in hl.split(",") if x.strip()
        )
        params["max_iter"] = st.number_input(
            "MLP epochs (max_iter)", 50, 2000, 300, 50, key="mlp_ep"
        )
        params["random_state"] = 42
    # LinearRegression has no key params here
    return params


def _multimodel_param_ui(selected_models: list[str]) -> dict[str, dict]:
    blocks: dict[str, dict] = {}
    with st.container(border=True):
        st.markdown("#### Hyperparameters (per selected model)")
        for m in selected_models:
            st.markdown(f"**{m}**")
            blocks[m] = _model_param_ui(m)
    return blocks


def _choose_display_pred(base_out, stacked, wgt, avg):
    # First Priority: Stacked ensemble
    if isinstance(stacked, dict) and stacked.get("pred") is not None:
        pred_output = stacked["pred"], "stacked_ensemble"
    # Second Priority: Weighted ensemble
    elif isinstance(wgt, dict) and wgt.get("pred") is not None:
        pred_output = wgt["pred"], "weighted_ensemble"
    # Third Priority: Simple Average ensemble
    elif isinstance(avg, dict) and avg.get("pred") is not None:
        pred_output = avg["pred"], "average_ensemble"
    # Fallback: single best model based on RMSE
    else:
        metrics = base_out["per_model_metrics"]
        best = min(metrics, key=lambda r: r["RMSE"])["model"]
        pred_output = base_out["valid_preds"][best], f"best_model:{best}"

    return pred_output


def _format_val(v):
    # format ints / floats; use scientific for very small p-values
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:d}"
    if isinstance(v, float):
        if 0 < v < 1e-4:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def _standardize_metrics_dict(d: dict) -> pd.DataFrame:
    """Turn a flat metrics dict into a 2-col DataFrame with friendly names."""
    if not isinstance(d, dict):
        return pd.DataFrame()
    rows = []
    for k, v in d.items():
        label = METRIC_NAMES.get(k, k)
        rows.append((label, _format_val(v)))
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df.set_index("Metric", inplace=True)
    return df


def _split_bp_test_results_from_model_metrics(model_metrics: list):
    """
    Inspect per_model metrics and, ONLY FOR DISPLAY:
    - find LR row (model name contains 'linear', case-insensitive),
    - extract BP (nested 'bp' OR flat keys),
    - return (cleaned_per_model_without_bp, bp_obj or None).
    """
    if not isinstance(model_metrics, list) or not model_metrics:
        return model_metrics, None

    cleaned: list[dict] = []
    bp_found = None

    for row in model_metrics:
        if not isinstance(row, dict):
            cleaned.append(row)
            continue

        row_copy = deepcopy(row)
        model_name = str(row_copy.get("model", "")).lower()
        is_lr = "linearregression" in model_name

        # Pull out bp from LR row only
        bp_val = row_copy.pop("bp", None) if is_lr and "bp" in row_copy else None

        if is_lr and bp_val is not None:
            parsed_bp = None

            # Case 1: already a dict
            if isinstance(bp_val, dict):
                parsed_bp = {str(k).lower(): v for k, v in bp_val.items()}

            # Case 2: stringified dict / tuple
            elif isinstance(bp_val, str):
                try:
                    tmp = ast.literal_eval(bp_val)
                except (ValueError, SyntaxError):
                    LOGGER.warning(
                        "Could not parse BP string: %s", bp_val, exc_info=True
                    )
                    tmp = None

                if isinstance(tmp, dict):
                    parsed_bp = {str(k).lower(): v for k, v in tmp.items()}
                elif isinstance(tmp, (list, tuple)) and len(tmp) >= 4:
                    # assume (lm, lm_pvalue, f, f_pvalue)
                    keys = ["lm", "lm_pvalue", "f", "f_pvalue"]
                    parsed_bp = {k: tmp[i] for i, k in enumerate(keys)}

            # Case 3: already a tuple/list of 4
            elif isinstance(bp_val, (list, tuple)) and len(bp_val) >= 4:
                keys = ["lm", "lm_pvalue", "f", "f_pvalue"]
                parsed_bp = {k: bp_val[i] for i, k in enumerate(keys)}

            if parsed_bp:
                bp_found = parsed_bp

        cleaned.append(row_copy)

    LOGGER.debug("BP object extracted [%s]", bp_found)
    return cleaned, bp_found


def _convert_bp_test_results_to_df(bp_obj) -> pd.DataFrame:
    """
    Normalize BP outputs into a 2-col table.
    Accepts:
      - dict with keys like: lm_stat / lm_pvalue / fvalue / f_pvalue
      - sequence/tuple of 4 (lm_stat, lm_pvalue, fvalue, f_pvalue)
      - any mixed naming.
    """
    LOGGER.info(f"BP output [{bp_obj}]]")
    # extract four numbers robustly
    lm = lmp = fstat = fp = None
    if isinstance(bp_obj, dict):
        # try common keys
        lm = bp_obj.get("lm")
        lmp = bp_obj.get("lm_pvalue")
        fstat = bp_obj.get("f")
        fp = bp_obj.get("f_pvalue")
    elif isinstance(bp_obj, (list, tuple)) and len(bp_obj) >= 4:
        lm, lmp, fstat, fp = bp_obj[:4]

    rows = [
        ("LM statistic", _format_val(lm) if lm is not None else "—"),
        ("LM p-value", _format_val(lmp) if lmp is not None else "—"),
        ("F statistic", _format_val(fstat) if fstat is not None else "—"),
        ("F p-value", _format_val(fp) if fp is not None else "—"),
    ]
    df = pd.DataFrame(rows, columns=["Breusch–Pagan", "Value"])
    df.set_index("Breusch–Pagan", inplace=True)
    return df


def _convert_model_metrics_to_df(model_metrics: list) -> pd.DataFrame:
    """Make per-model metrics pretty with expanded names and numeric formatting."""
    if not model_metrics:
        return pd.DataFrame()
    df = pd.DataFrame(model_metrics).copy()

    # rename columns if present
    rename_map = {k: v for k, v in METRIC_NAMES.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # nice float formatting for known columns
    for col in df.columns:
        df[col] = df[col].map(_format_val)

    # Remove 'bp' from display
    if "bp" in df.columns:
        df = df.drop(columns=["bp"])

    # put 'model' as index if present
    if "model" in df.columns:
        df.set_index("model", inplace=True)

    return df


# ----------------------------------------------------------------------
# Visual diagnostics (migrated from Visual Tools into Modeling)
# ----------------------------------------------------------------------
def _vt_get_truth_pred_from_last_model(res):
    """
    Try to retrieve y_true / y_pred from last_model payload.
    Primary: top-level keys; fallback: base['y_true'/'y_pred'].
    """
    if not isinstance(res, dict):
        return None, None

    if "y_true" in res and "y_pred" in res:
        return res["y_true"], res["y_pred"]

    base = res.get("base") or {}
    return base.get("y_true"), base.get("y_pred")


def _vt_display_chart(img_data_uri: str):
    """
    Handle both data URI and raw bytes images coming back from visual_tools_service.
    """
    if isinstance(img_data_uri, str) and img_data_uri.startswith("data:image"):
        # data:image/png;base64,....
        try:
            b64 = img_data_uri.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            st.image(img_bytes, use_column_width=True)
        except Exception as exc:
            LOGGER.warning("Failed to decode base64 image: %s", exc)
            st.image(img_data_uri, use_column_width=True)
    else:
        st.image(img_data_uri, use_column_width=True)


def _render_visual_diagnostics(res: dict):
    """
    Render Actual vs Predicted and Residual diagnostics under Modeling,
    using the same charts that were previously in the Visual Tools step.
    """
    y_true, y_pred = _vt_get_truth_pred_from_last_model(res)
    if y_true is None or y_pred is None:
        return

    st.markdown("### Visual Diagnostics")

    p1 = chart_actual_vs_pred(y_true, y_pred)
    p2 = chart_residuals(y_true, y_pred)
    p3 = chart_residuals_qq(y_true, y_pred)

    # 2 charts on top row
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.caption("Actual vs Predicted")
        _vt_display_chart(p1)
    with c2:
        st.caption("Residuals vs Fitted")
        _vt_display_chart(p2)

    # 1 chart on second row
    c3, _ = st.columns(2, gap="medium")
    with c3:
        st.caption("Residuals Q–Q Plot")
        _vt_display_chart(p3)


def _show_last_model_stats():
    """Bottom panel with stats + visual diagnostics from session."""
    res = st.session_state.get("last_model") or {}
    if res:
        st.divider()
        st.markdown("### Last Model Stats")

        # --- Ensembles: Stacked / Weighted / Average ---
        col1, col2, col3 = st.columns(3)

        # Stacked Ensemble
        with col1:
            st.markdown("**Ensemble: Stacked (meta-model)**")
            stacked_metrics = (res.get("ensemble_stacked") or {}).get("metrics")
            if stacked_metrics:
                st.dataframe(
                    _standardize_metrics_dict(stacked_metrics),
                    use_container_width=True,
                )
            else:
                st.info("No stacked ensemble metrics found.")

        # Weighted Ensemble
        with col2:
            st.markdown("**Ensemble: Weighted (1/RMSE)**")
            wgt_metrics = (res.get("ensemble_wgt") or {}).get("metrics")
            if wgt_metrics:
                st.dataframe(
                    _standardize_metrics_dict(wgt_metrics),
                    use_container_width=True,
                )
            else:
                st.info("No weighted ensemble metrics found.")

        # Simple Average Ensemble
        with col3:
            st.markdown("**Ensemble: Simple Average**")
            avg_metrics = (res.get("ensemble_avg") or {}).get("metrics")
            if avg_metrics:
                st.dataframe(
                    _standardize_metrics_dict(avg_metrics),
                    use_container_width=True,
                )
            else:
                st.info("No simple average metrics found.")

        # --- Per-model metrics (expanded names) + BP extraction ---
        st.markdown("**Model Validation Metrics**")
        model_metrics = (res.get("base") or {}).get("per_model_metrics") or []
        model_metrics, bp_results = _split_bp_test_results_from_model_metrics(
            model_metrics
        )
        model_metrics = _convert_model_metrics_to_df(model_metrics)
        st.dataframe(model_metrics)

        # --- Breusch–Pagan (from LR row) ---
        if bp_results is not None:
            st.markdown("**Heteroscedasticity (Breusch–Pagan)**")
            st.dataframe(_convert_bp_test_results_to_df(bp_results))

        pred_src = res.get("pred_source") or "unknown"
        st.caption(
            f"**Prediction source used for visuals**: "
            f"{PRED_SRC_DISP_NAMES.get(pred_src, pred_src)}"
        )

        # --- Visual diagnostics (charts from former Visual Tools step) ---
        _render_visual_diagnostics(res)

        run_dir = st.session_state.get("last_model_run_dir")
        if run_dir:
            st.caption(f"**Artifacts saved under**: `{run_dir}`")


def _render_model_registry_panel(run_id: str) -> None:
    """
    Show registry info for current model (if any), and allow registration.
    """
    last_model = st.session_state.get("last_model") or {}
    if not last_model:
        # Nothing trained/activated yet → no registry UI
        return

    # Model name stored in store_last_model_info_in_session
    model_name = last_model.get("model_name")
    if not model_name:
        model_name = f"ppe_model_{run_id}"

    # Load current registry index
    try:
        registry_entries = load_model_registry()
    except Exception as exc:
        LOGGER.warning("Could not load model registry: %s", exc)
        st.info("Model registry not available at the moment.")
        return

    # Check if there is already an entry for (run_id, model_name)
    existing_entry = next(
        (
            entry
            for entry in registry_entries
            if entry.get("run_id") == run_id
               and entry.get("model_name") == model_name
        ),
        None,
    )

    st.divider()
    with st.container(border=True):
        st.markdown("### Model Registry")

        st.write(f"**Model:** `{model_name}`")
        st.write(f"**Pipeline run:** `{run_id}`")

        if existing_entry:
            status = existing_entry.get("status", "unknown")
            created_at = existing_entry.get("created_at")
            created_by = existing_entry.get("created_by")

            st.success(
                f"This model is already registered with status **{status}**."
            )

            meta_bits = []
            if created_at:
                meta_bits.append(f"created at `{created_at}`")
            if created_by:
                meta_bits.append(f"by `{created_by}`")
            if meta_bits:
                st.caption(", ".join(meta_bits))
        else:
            st.info("This model is not yet registered in the global registry.")

            if st.button("Register this model", key="btn_register_model"):
                username = st.session_state.get("username", "unknown")
                register_model_in_registry(
                    run_id=run_id,
                    model_name=model_name,
                    status="candidate",
                    created_by=username,
                )
                st.success(
                    f"Model `{model_name}` registered as **candidate** "
                    f"for this pipeline run."
                )


# ----------------------------------------------------------------------
# Page render (MENU-TRIGGERED ENTRY POINT)
# ----------------------------------------------------------------------
@streamlit_safe
def render():
    st.header("Modeling")
    show_last_training_badge()

    run_id = st.session_state.run_id
    df = st.session_state.cleaned_df
    if df is None:
        LOGGER.warning("No cleaned feature master available in session")
        return

    label = os.path.basename(Path(st.session_state.last_cleaned_feature_master_path))
    st.caption(f"Using: {label} — shape={df.shape}")

    num_cols = _numeric_columns(df)
    if not num_cols:
        st.error("No numeric columns found in the selected feature master.")
        return

    # Model Name
    default_model_name = f"ppe_model_{run_id}"
    with st.container(border=True):
        st.markdown("#### Model Name")

        model_name_input = st.text_input(
            "Model name (optional)",
            value="",
            placeholder=default_model_name,
            help=(
                "If left blank, the model will be saved as "
                f"`{default_model_name}`."
            ),
            key="model_name",
        )

    # Final model name to be used for the rest of the pipeline
    model_name = model_name_input or default_model_name

    # Target + features
    with st.container(border=True):
        st.markdown("#### Feature Selection")
        default_target = "price"
        target = st.selectbox(
            "Target (dependent variable)",
            num_cols,
            index=num_cols.index(default_target) if default_target in num_cols else 0,
        )
        features = st.multiselect(
            "Feature columns (independent variables)",
            [c for c in df.columns if c != target],
            default=[c for c in num_cols if c != target],
        )
        if not features:
            st.info("Please select at least one feature.")
            return

    # Allowed models only
    allowed = AVAILABLE_MODELS.keys()
    with st.container(border=True):
        st.markdown("#### Model Selection")
        selected_models = st.multiselect(
            "Select models to train",
            allowed,
            default=["LinearRegression", "LightGBM", "MLP"],
        )
    if not selected_models:
        st.info("Select at least one model.")
        return

    _compare_model_selection_from_last_run(selected_models)

    params_map = _multimodel_param_ui(selected_models)

    # Train model & Save artifacts
    if st.button("Train", type="primary"):
        with st.spinner(f"Training on {label}..."):
            base_train_out = train_base_models(
                df[features + [target]],
                target,
                selected_models,
                params_map,
            )

        st.subheader("Per-model Validation Metrics")
        st.dataframe(
            pd.DataFrame(base_train_out["per_model_metrics"]).set_index("model")
        )

        # Ensembles (computed automatically behind the scenes)
        stacked = build_stacked_ensemble(base_train_out)
        avg = combine_average(base_train_out)
        wgt = combine_weighted_inverse_rmse(base_train_out)

        # Result predictions (prefer ensembles)
        y_valid = base_train_out["y_valid"]
        y_true = y_valid  # minimize other changes
        y_pred, pred_src = _choose_display_pred(base_train_out, stacked, wgt, avg)

        st.subheader("Ensemble (automatic)")
        st.markdown("**Stacked Ensemble (meta-model on base predictions)**")
        st.json(stacked["metrics"])

        st.markdown("**Simple Average (for comparison)**")
        st.json(avg["metrics"])

        st.markdown("**Weighted Average (by inverse RMSE, for comparison)**")
        st.json(wgt["metrics"])

        # ----------------------------
        # model + X_valid + X_sample for explainability
        # ----------------------------
        x_valid = base_train_out.get("X_valid")
        model_for_explain = None
        x_sample = None

        # Pick the best base model by RMSE (same logic as before, but explicit here)
        per_model_metrics = base_train_out.get("per_model_metrics") or []
        best_model_name = None
        if per_model_metrics:
            try:
                best_row = min(per_model_metrics, key=lambda r: r["RMSE"])
                best_model_name = str(best_row.get("model"))
            except Exception as ex:  # defensive, never break training
                LOGGER.warning(f"Could not determine best model for explainability: {ex}")

        # Try to grab the fitted model object from the training output
        models_dict = base_train_out.get("models") or base_train_out.get("fitted_models") or {}
        if best_model_name and isinstance(models_dict, dict):
            model_for_explain = models_dict.get(best_model_name)

        # Sample a smaller subset for SHAP to keep it tractable
        if x_valid is not None:
            try:
                x_sample = x_valid.sample(
                    n=min(200, len(x_valid)),
                    random_state=42,
                )
            except Exception as ex:
                LOGGER.warning(f"Unable to sample X_valid for SHAP: {ex}")
                x_sample = None

        # ----------------------------
        # Persist context (artifacts + session state)
        # ----------------------------
        artifact_location = save_model_artifacts(
            run_id=run_id,
            base_out=base_train_out,
            stacked=stacked,
            comb_avg=avg,
            comb_wgt_inv_rmse=wgt,
            params_map=params_map,
            y_true=y_true,
            y_pred=y_pred,
            pred_src=pred_src,
            model_name=model_name,
            x_valid=x_valid,
            y_valid=y_valid,
            x_sample=x_sample,
        )
        st.session_state["model_trained"] = True

        # NOTE: new arguments at the tail: model_for_explain, X_valid, y_valid, X_sample
        store_last_model_info_in_session(
            base_train_out,
            stacked,
            avg,
            wgt,
            y_true,
            y_pred,
            pred_src,
            params_map,
            selected_models,
            model_name=model_for_explain,
            x_valid=x_valid,
            y_valid=y_valid,
            x_sample=x_sample,
        )

        store_last_run_model_dir_in_session(artifact_location)

        st.success(f"Model {model_name} trained and saved to {artifact_location}")

    # Always show latest stats & diagnostics if a last_model is present
    _show_last_model_stats()

    # Registry panel for the active model / run
    _render_model_registry_panel(run_id)
