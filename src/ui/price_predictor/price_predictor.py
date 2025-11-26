from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Final

import numpy as np
import pandas as pd
import streamlit as st

from src.config.config_loader import load_config_from_file
from src.services.price_predictor.price_predictor_service import (
    RegistryModelOption,
    get_registry_model_options,
    load_stacked_model,
    predict_for_dataframe,
    predict_for_single_asset,
    save_uploaded_assets_df,
    compute_confidence_interval_and_score, save_batch_prediction_audit, save_single_prediction_audit,
)

_ASSET_INPUT_LBL_SINGLE: Final[str] = "Enter asset manually"
_ASSET_INPUT_LBL_MULTI: Final[str] = "Upload asset file"


# -------------------------------------------------------------------
# Helper functions – input / forms
# -------------------------------------------------------------------
def _load_predictor_option_lists() -> Dict[str, dict]:
    """
    Load ui_config.toml using the global config loader and extract ONLY:
      - territories
      - medias
      - platforms
      - license_types

    Each entry is a mapping CODE -> LABEL.
    """
    try:
        cfg = load_config_from_file("src/config/ui_config.toml")
    except Exception:
        return {
            "territories": {},
            "medias": {},
            "platforms": {},
            "license_types": {},
        }

    def get_map(key: str) -> dict:
        val = cfg.get(key)
        return val if isinstance(val, dict) else {}

    return {
        "territories": get_map("territories"),
        "medias": get_map("medias"),
        "platforms": get_map("platforms"),
        "license_types": get_map("license_types"),
    }


def _render_manual_asset_form() -> Dict[str, Any]:
    """
    Manual entry uses mappings defined in ui_config.toml:

        territories, medias, platforms, license_types

    The dropdown shows the VALUE (label), but the returned data uses the KEY
    (e.g., "USA", "SVOD", "NETFLIX", "EXCLUSIVE").
    """
    option_maps = _load_predictor_option_lists()

    territories_map = option_maps["territories"]
    medias_map = option_maps["medias"]
    platforms_map = option_maps["platforms"]
    license_types_map = option_maps["license_types"]

    # Fallbacks if config is missing something
    if not territories_map:
        territories_map = {
            "USA": "United States of America",
            "GBR": "United Kingdom",
        }
    if not medias_map:
        medias_map = {
            "SVOD": "Subscription Video on-demand",
            "TVOD": "Transaction Video on-demand",
        }
    if not platforms_map:
        platforms_map = {
            "GENERIC": "Generic Platform",
        }
    if not license_types_map:
        license_types_map = {
            "EXCLUSIVE": "Exclusive",
            "NON_EXCLUSIVE": "Non-Exclusive",
        }

    # Build display lists and reverse maps (label -> key)
    territory_labels = list(territories_map.values())
    territory_label_to_key = {v: k for k, v in territories_map.items()}

    media_labels = list(medias_map.values())
    media_label_to_key = {v: k for k, v in medias_map.items()}

    platform_labels = list(platforms_map.values())
    platform_label_to_key = {v: k for k, v in platforms_map.items()}

    license_type_labels = list(license_types_map.values())
    license_type_label_to_key = {v: k for k, v in license_types_map.items()}

    # --- UI fields (display labels) ---
    territory_label = st.selectbox("Territory", options=territory_labels, index=0)

    col1, col2 = st.columns(2)
    with col1:
        media_label = st.selectbox("Media", media_labels)
        platform_label = st.selectbox("Platform", platform_labels)
    with col2:
        license_type_label = st.selectbox("License Type", license_type_labels)
        window_months = st.number_input("window_months", min_value=1, max_value=120, value=24)

    col3, col4 = st.columns(2)
    with col3:
        release_year = st.number_input("release_year", min_value=1970, max_value=2100, value=2020)
        imdb_rating = st.slider("imdb_rating", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    with col4:
        imdb_votes = st.number_input("imdb_votes", min_value=0, value=10000, step=100)
        popularity_index = st.slider(
            "popularity_index", min_value=0.0, max_value=1.0, value=0.6, step=0.01
        )

    # Convert selected labels back to their keys
    territory_key = territory_label_to_key[territory_label]
    media_key = media_label_to_key[media_label]
    platform_key = platform_label_to_key[platform_label]
    license_type_key = license_type_label_to_key[license_type_label]

    # NOTE: returning the *keys* here
    return {
        "territory": territory_key,
        "media": media_key,
        "platform": platform_key,
        "exclusivity": license_type_key,
        "window_months": int(window_months),
        "release_year": int(release_year),
        "imdb_rating": float(imdb_rating),
        "imdb_votes": int(imdb_votes),
        "popularity_index": float(popularity_index),
    }


def _handle_file_upload() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Handle the file upload for multi-asset prediction mode.

    Returns:
        (uploaded_df, upload_name)
    """
    uploaded_df: Optional[pd.DataFrame] = None
    upload_name: Optional[str] = None

    st.markdown(
        "Upload a CSV or Parquet file with **one row per asset** and either "
        "raw metadata columns (territory, media, platform, imdb_votes, etc.) "
        "or the full feature-master schema the model was trained on."
    )
    uploaded_file = st.file_uploader(
        "Upload asset dataset",
        type=["csv", "parquet"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return None, None

    upload_name = uploaded_file.name
    try:
        if upload_name.lower().endswith(".parquet"):
            uploaded_df = pd.read_parquet(uploaded_file)
        else:
            uploaded_df = pd.read_csv(uploaded_file)

        st.success(
            f"Loaded {len(uploaded_df):,} records from `{upload_name}` "
            f"with {len(uploaded_df.columns)} columns."
        )
        st.markdown("#### Preview of uploaded dataset")
        st.dataframe(uploaded_df.head(50))

    except Exception as ex:  # noqa: BLE001
        st.error(f"Error reading uploaded file: {ex}")
        uploaded_df = None
        upload_name = None

    return uploaded_df, upload_name


def _render_inputs(
        options: list[RegistryModelOption],
) -> Tuple[
    RegistryModelOption,
    str,
    Optional[pd.DataFrame],
    Optional[Dict[str, Any]],
    Optional[str],
    bool,
]:
    """
    Render the top expander with:

    - model dropdown
    - prediction mode radio
    - file upload OR manual form
    - predict button
    """
    with st.expander("Asset Input", expanded=True):
        input_mode = st.radio(
            "Asset Selection",
            options=[
                _ASSET_INPUT_LBL_MULTI,
                _ASSET_INPUT_LBL_SINGLE,
            ],
            key="predict_price_input_mode",
        )

        labels = [opt.label for opt in options]
        selected_label = st.selectbox(
            "Select model from registry",
            options=labels,
            index=0,  # options are pre-sorted DESC by created_at
        )
        selected_option = options[labels.index(selected_label)]

        # Format date as MON DD, YYYY (e.g., JAN 05, 2025)
        created_str = selected_option.created_at.strftime("%b %d, %Y").upper()
        st.caption(
            f"Using model **{selected_option.model_name}** "
            f"(Pipeline Run ID=`{selected_option.run_id}`, "
            f"created on {created_str})."
        )

        st.markdown("---")

        uploaded_df: Optional[pd.DataFrame] = None
        manual_asset: Optional[Dict[str, Any]] = None
        upload_name: Optional[str] = None

        if input_mode == _ASSET_INPUT_LBL_MULTI:
            uploaded_df, upload_name = _handle_file_upload()
        else:
            st.markdown(
                "Enter feature values for a single media asset. "
                "Field names should align with the model's expected feature columns."
            )
            manual_asset = _render_manual_asset_form()

        submitted = st.button("Run prediction", type="primary")

    return selected_option, input_mode, uploaded_df, manual_asset, upload_name, submitted


# -------------------------------------------------------------------
# Helper functions – results
# -------------------------------------------------------------------
def _render_single_asset_result(
        price: float,
        df_single: pd.DataFrame,
        asset_features: Optional[Dict[str, Any]],
        registry_option: RegistryModelOption,
        ci_low: Optional[float],
        ci_high: Optional[float],
        confidence_pct: Optional[float],
) -> None:
    """
    Render rich single-asset view:

      1. Price card
      2. CI / confidence / model version row
      3. Top active features bar chart
      4. RMSE-based error band info
      5. Raw asset metadata
    """
    # 1. Price card
    st.markdown(
        f"""
        <div class="ppe-price-card">
            <div class="ppe-price-label">Predicted licensing price</div>
            <div class="ppe-price-value">${price:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. CI / confidence / model version
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Confidence interval (approx. 95%)**")
        if ci_low is not None and ci_high is not None:
            st.markdown(f"${ci_low:,.0f} – ${ci_high:,.0f}")
        else:
            st.markdown("_Not available_")

    with col2:
        st.markdown("**Prediction confidence**")
        if confidence_pct is not None:
            st.markdown(f"{confidence_pct:0.0f}%")
        else:
            st.markdown("_Not available_")

    with col3:
        st.markdown("**Model version**")
        model_label = registry_option.model_name or "N/A"
        run_id = registry_option.run_id or "N/A"
        st.markdown(f"`{model_label}`  \nPipeline Run ID: `{run_id}`")

    st.markdown("---")

    # 3. Top active features bar chart (from engineered row)
    st.markdown("#### Top active features for this prediction")
    if not df_single.empty:
        row = df_single.iloc[0].drop(labels=["predicted_price"], errors="ignore")
        used = row[(row != 0) & (~row.isna())]
        if not used.empty:
            used = used.reindex(used.abs().sort_values(ascending=False).index).head(10)
            feature_df = (
                pd.DataFrame(
                    {"Feature": used.index.astype(str), "Value": used.values.astype(float)}
                )
                .set_index("Feature")
            )
            st.bar_chart(feature_df)
        else:
            st.caption("No non-zero features found for this prediction row.")
    else:
        st.caption("No feature data available for this prediction.")

    # 4. RMSE / error band info
    metrics = registry_option.metrics or {}
    rmse = None
    for key in ("rmse", "RMSE", "val_rmse"):
        if key in metrics:
            try:
                rmse = float(metrics[key])
                break
            except (TypeError, ValueError):
                continue

    if rmse is not None and rmse > 0:
        band_low = price - 2 * rmse
        band_high = price + 2 * rmse
        st.info(
            f"For assets similar to this one, the model's RMSE is approximately "
            f"${rmse:,.0f}. A typical error band around this prediction is from "
            f"${band_low:,.0f} to ${band_high:,.0f}."
        )

    # 5. Raw asset metadata
    if asset_features:
        st.markdown("#### Asset metadata used for this prediction")
        st.json(asset_features)


def _render_multi_asset_results(
        pred_df: pd.DataFrame,
        registry_option: RegistryModelOption,
        ci_low: Optional[float],
        ci_high: Optional[float],
        confidence_pct: Optional[float],
) -> None:
    """
    Render batch view with layout parallel to single-asset:

      1. Price card (average price)
      2. CI / confidence / model version row
      3. Price distribution chart
      4. RMSE-based error band info
      5. Predictions table
    """
    if "predicted_price" not in pred_df.columns:
        st.error("Prediction DataFrame does not have a 'predicted_price' column.")
        return

    n_rows = len(pred_df)
    price_series = pred_df["predicted_price"].astype(float)
    avg_price = float(price_series.mean())

    # 1. Price card (average price)
    st.markdown(
        f"""
        <div class="ppe-price-card">
            <div class="ppe-price-label">Average predicted licensing price</div>
            <div class="ppe-price-value">${avg_price:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. CI / confidence / model version (same structure as single)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Confidence interval (approx. 95%)**")
        if ci_low is not None and ci_high is not None:
            st.markdown(f"${ci_low:,.0f} – ${ci_high:,.0f}")
        else:
            st.markdown("_Not available_")

    with col2:
        st.markdown("**Prediction confidence**")
        if confidence_pct is not None:
            st.markdown(f"{confidence_pct:0.0f}%")
        else:
            st.markdown("_Not available_")

    with col3:
        st.markdown("**Model version**")
        model_label = registry_option.model_name or "N/A"
        run_id = registry_option.run_id or "N/A"
        st.markdown(f"`{model_label}`  \nPipeline Run ID: `{run_id}`")

    st.markdown("---")

    # 3. Price distribution chart ("graph")
    st.markdown("#### Predicted price distribution for this batch")
    if n_rows > 0:
        # Build a simple histogram as bar chart
        hist, bin_edges = np.histogram(price_series, bins=min(10, max(1, n_rows)))
        bin_labels = [
            f"{bin_edges[i]:,.0f}–{bin_edges[i + 1]:,.0f}"
            for i in range(len(bin_edges) - 1)
        ]
        hist_df = pd.DataFrame({"Price bin": bin_labels, "Count": hist}).set_index(
            "Price bin"
        )
        st.bar_chart(hist_df)
    else:
        st.caption("No predictions available for this batch.")

    # 4. RMSE / error band info around the average price
    metrics = registry_option.metrics or {}
    rmse = None
    for key in ("rmse", "RMSE", "val_rmse"):
        if key in metrics:
            try:
                rmse = float(metrics[key])
                break
            except (TypeError, ValueError):
                continue

    if rmse is not None and rmse > 0:
        band_low = avg_price - 2 * rmse
        band_high = avg_price + 2 * rmse
        st.info(
            f"For batches of assets similar to this one, the model's RMSE is approximately "
            f"${rmse:,.0f}. A typical error band around the average prediction is from "
            f"${band_low:,.0f} to ${band_high:,.0f}."
        )

    # 5. Predictions table
    st.markdown("#### Predictions table")
    st.dataframe(pred_df)


def _run_multi_asset_prediction(
        model: Any,
        uploaded_df: Optional[pd.DataFrame],
        upload_name: Optional[str],
        registry_option: RegistryModelOption,
) -> None:
    """
    Execute and render multi-asset predictions and persist an audit record.
    """
    if uploaded_df is None:
        st.error("No valid dataset uploaded. Please upload a file and try again.")
        return

    input_ref: Optional[str] = None
    if upload_name:
        try:
            input_ref = save_uploaded_assets_df(uploaded_df, upload_name)
            st.caption(f"Uploaded dataset saved for audit at: `{input_ref}`")
        except Exception as ex:  # noqa: BLE001
            st.warning(f"Could not persist uploaded dataset: {ex}")

    try:
        pred_df = predict_for_dataframe(model, uploaded_df)
    except Exception as ex:  # noqa: BLE001
        st.error(f"Prediction failed for uploaded dataset: {ex}")
        return

    # Use average price for CI / confidence
    price_series = pred_df["predicted_price"].astype(float)
    avg_price = float(price_series.mean())
    ci_low, ci_high, conf_pct = compute_confidence_interval_and_score(
        avg_price,
        registry_option.metrics,
    )

    # Persist batch audit (full pred_df + JSON summary)
    try:
        audit_refs = save_batch_prediction_audit(
            option=registry_option,
            pred_df=pred_df,
            input_ref=input_ref,
            ci_low=ci_low,
            ci_high=ci_high,
            confidence_pct=conf_pct,
        )
        st.caption(
            f"Batch prediction run saved for reporting "
            f"(data: `{audit_refs.get('data_ref', '')}`, "
            f"meta: `{audit_refs.get('meta_ref', '')}`)."
        )
    except Exception as ex:  # noqa: BLE001
        st.warning(f"Could not persist batch prediction audit: {ex}")

    _render_multi_asset_results(pred_df, registry_option, ci_low, ci_high, conf_pct)


def _run_single_asset_prediction(
        model: Any,
        manual_asset: Optional[Dict[str, Any]],
        registry_option: RegistryModelOption,
) -> None:
    """
    Execute and render single-asset prediction and persist an audit record.
    """
    if manual_asset is None:
        st.error("No asset metadata provided. Please fill in the form and try again.")
        return

    try:
        price, df_single = predict_for_single_asset(model, manual_asset)
    except Exception as ex:  # noqa: BLE001
        st.error(f"Prediction failed for the entered asset: {ex}")
        return

    ci_low, ci_high, conf_pct = compute_confidence_interval_and_score(
        price,
        registry_option.metrics,
    )

    # Persist audit (Parquet + JSON in predictions dir/bucket)
    try:
        audit_refs = save_single_prediction_audit(
            option=registry_option,
            asset_features=manual_asset,
            price=price,
            ci_low=ci_low,
            ci_high=ci_high,
            confidence_pct=conf_pct,
        )
        st.caption(
            f"Prediction run saved for reporting "
            f"(data: `{audit_refs.get('data_ref', '')}`, "
            f"meta: `{audit_refs.get('meta_ref', '')}`)."
        )
    except Exception as ex:  # noqa: BLE001
        st.warning(f"Could not persist prediction audit: {ex}")

    _render_single_asset_result(
        price=price,
        df_single=df_single,
        asset_features=manual_asset,
        registry_option=registry_option,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_pct=conf_pct,
    )


# -------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------
def render_price_predictor() -> None:
    """
    Predict Licensing Price screen.

    - Uses global model registry (model_registry.json under MODELS_DIR / MODELS_BUCKET).
    - Does NOT depend on any run_id or RunInfo.
    - Layout: top expander for model + inputs, bottom *expander* for results.
    """
    st.markdown("## Predict Licensing Price")

    # 1. Load registry entries
    options = get_registry_model_options(status_filter=None)
    if not options:
        st.warning(
            "No models found in the registry. Train and register at least one model "
            "before using the Predict Price screen."
        )
        return

    # 2. Render inputs & collect context
    (
        selected_option,
        input_mode,
        uploaded_df,
        manual_asset,
        upload_name,
        submitted,
    ) = _render_inputs(options)

    # 3. Results section in an expander
    st.markdown("---")
    with st.expander("Prediction results", expanded=True):
        if not submitted:
            st.caption("Predicted prices and details will appear here after you run a prediction.")
            return

        # 4. Load model and run prediction
        with st.spinner("Loading model and running predictions..."):
            try:
                model = load_stacked_model(selected_option)
            except Exception as ex:  # noqa: BLE001
                st.error(
                    f"Failed to load model `{selected_option.model_name}` "
                    f"for Pipeline Run ID `{selected_option.run_id}`: {ex}"
                )
                return

            if input_mode == _ASSET_INPUT_LBL_MULTI:
                _run_multi_asset_prediction(
                    model, uploaded_df, upload_name, selected_option
                )
            else:
                _run_single_asset_prediction(model, manual_asset, selected_option)
