from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st
import pandas as pd

from src.services.compare_and_bundling.bundling_service import (
    get_available_strategies,
    generate_candidate_bundles,
)
from src.services.compare_and_bundling.comparison_service import build_compare_table
from src.services.price_predictor.price_predictor_service import (
    RegistryModelOption,
    get_registry_model_options,
    load_stacked_model,
    predict_for_dataframe,
)
from src.utils.data_io_utils import (
    save_predictions,
    save_prediction_metadata,
)


# ---------- Title input & options helpers ----------------------------------


def _load_titles_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load the uploaded titles file into a DataFrame.
    Currently supports CSV; you can extend this to parquet etc. later.
    """
    if uploaded_file is None:
        return None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Failed to read uploaded file. Please upload a valid CSV.")
        return None

    if df.empty:
        st.warning("Uploaded file is empty.")
        return None

    return df


def _get_title_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Guess the ID column from common patterns.
    """
    for col in ["title_id", "tconst", "id"]:
        if col in df.columns:
            return col
    return None


def _get_title_name_value(row: pd.Series) -> str:
    for col in ["title_name", "primary_title", "original_title", "title"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return "Unknown Title"


def _build_title_label(row: pd.Series) -> str:
    """
    Build label like:
      Title Name - Territory - Media - Platform - LicenseType
    Only includes fields that are present and non-null.
    """
    parts = [_get_title_name_value(row)]

    # Territory / region
    for col in ["territory", "region", "country"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break

    # Media
    if "media" in row.index and pd.notna(row["media"]):
        parts.append(str(row["media"]))

    # Platform
    if "platform" in row.index and pd.notna(row["platform"]):
        parts.append(str(row["platform"]))

    # License type
    for col in ["license_type", "license", "license_category"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break

    return " - ".join(parts)


def _build_title_options_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping: title_id -> label for dropdowns,
    using the uploaded titles DataFrame.
    """
    id_col = _get_title_id_column(df)
    if id_col is None:
        st.error(
            "Could not determine title ID column. "
            "Expected one of: 'title_id', 'tconst', or 'id'."
        )
        return {}

    options: Dict[str, str] = {}
    for _, row in df.iterrows():
        title_id = str(row[id_col])
        label = _build_title_label(row)
        options[title_id] = label

    return options


# ---------- Model selection helpers ----------------------------------------

def _select_registry_model() -> Optional[RegistryModelOption]:
    """
    Render a registry model selectbox and return the chosen RegistryModelOption,
    or None if no models are available.
    """
    options = get_registry_model_options(status_filter=None)
    if not options:
        st.warning("No models available in registry. Train and register a model first.")
        return None

    labels = [opt.label for opt in options]

    previous_label = st.session_state.get("compare_bundling_selected_model_label")
    default_index = labels.index(previous_label) if previous_label in labels else 0

    selected_label = st.selectbox(
        "Select model for predictions",
        options=labels,
        index=default_index,
    )

    st.session_state["compare_bundling_selected_model_label"] = selected_label

    for opt in options:
        if opt.label == selected_label:
            st.session_state["compare_bundling_selected_model"] = opt
            return opt

    return None


def _fetch_predictions_for_titles(
        selected_title_ids: List[str],
        titles_df: Optional[pd.DataFrame],
        model_option: Optional[RegistryModelOption],
) -> pd.DataFrame:
    """
    Use the selected registry model to predict prices for the given title IDs,
    using the uploaded titles_df as raw metadata.

    Returns a DataFrame with at least:
      - title_id
      - title_name
      - predicted_price
      - region (mapped from territory/country if needed)
      - platform
      - release_year
      - genres
    """
    if not selected_title_ids:
        return pd.DataFrame()
    if titles_df is None or titles_df.empty:
        st.warning("No titles data available; upload a titles file first.")
        return pd.DataFrame()
    if model_option is None:
        st.warning("No model selected; choose a model above before running predictions.")
        return pd.DataFrame()

    id_col = _get_title_id_column(titles_df)
    if id_col is None:
        st.error(
            "Could not determine title ID column in uploaded file. "
            "Expected one of: 'title_id', 'tconst', or 'id'."
        )
        return pd.DataFrame()

    mask = titles_df[id_col].astype(str).isin(selected_title_ids)
    df_raw = titles_df.loc[mask].copy()
    if df_raw.empty:
        st.warning("Selected titles not found in uploaded titles file.")
        return pd.DataFrame()

    # Load model and run predictions using the same feature-building logic
    # as the main price prediction service.
    model = load_stacked_model(model_option)
    preds_df = predict_for_dataframe(model, df_raw, price_column_name="predicted_price")

    # Ensure common columns expected by compare/bundling services.
    preds_df["title_id"] = df_raw[id_col].astype(str).values

    if "title_name" not in preds_df.columns:
        preds_df["title_name"] = df_raw.apply(_get_title_name_value, axis=1)

    if "region" not in preds_df.columns:
        if "territory" in df_raw.columns:
            preds_df["region"] = df_raw["territory"]
        elif "country" in df_raw.columns:
            preds_df["region"] = df_raw["country"]

    # platform, release_year, genres will already be present if they exist in df_raw.
    # We leave them as-is.

    return preds_df


# ---------- Persistence helpers (for Reports) -------------------------------

_COMPARE_BASE_DIR = "compare"
_BUNDLING_BASE_DIR = "bundling"


def _bundle_results_to_df(bundle_results: List[Any]) -> pd.DataFrame:
    """
    Flatten BundleResult objects into a row-level DataFrame.

    Columns:
      - bundle_id, strategy_name, bundle_* scores
      - title_id, title_name, predicted_price, region, platform, release_year, genres
    """
    rows: List[Dict[str, Any]] = []

    for i, bundle in enumerate(bundle_results, start=1):
        bundle_id = f"bundle_{i}"

        for t in bundle.titles:
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "strategy_name": bundle.strategy_name,
                    "bundle_price_raw": bundle.bundle_price_raw,
                    "bundle_value_score": bundle.bundle_price_score,
                    "bundle_fit_score": bundle.bundle_fit_score,
                    "bundle_diversity_score": bundle.bundle_diversity_score,
                    "bundle_risk_score": bundle.bundle_risk_score,
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


def _model_meta_dict(model_option: Optional[RegistryModelOption]) -> Optional[Dict[str, Any]]:
    if model_option is None:
        return None
    return {
        "run_id": model_option.run_id,
        "model_name": model_option.model_name,
        "status": model_option.status,
        "metrics": model_option.metrics or {},
    }


def _persist_compare_run(
        predictions_df: pd.DataFrame,
        selected_ids: List[str],
        selected_labels: List[str],
        model_option: Optional[RegistryModelOption],
) -> None:
    """Persist comparison predictions + metadata for later reports."""
    if predictions_df.empty:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"compare_{ts}"
    base_dir = _COMPARE_BASE_DIR  # "compare"

    try:
        predictions_uri = save_predictions(predictions_df, base_dir, name)
    except Exception as ex:
        st.warning(f"Unable to save comparison predictions: {ex}")
        return

    meta = {
        "module": "compare_and_bundling",
        "mode": "compare",
        "base_dir": base_dir,
        "artifact_name": name,
        "timestamp": ts,
        "selected_title_ids": selected_ids,
        "selected_title_labels": selected_labels,
        "rows": int(len(predictions_df)),
        "columns": list(predictions_df.columns),
        "predictions_uri": predictions_uri,
        "model": _model_meta_dict(model_option),
    }

    try:
        meta_uri = save_prediction_metadata(meta, base_dir, name)
    except Exception as ex:
        st.warning(f"Unable to save comparison metadata: {ex}")
        meta_uri = None

    st.session_state["compare_last_predictions_uri"] = predictions_uri
    st.session_state["compare_last_metadata_uri"] = meta_uri


def _persist_bundling_run(
        bundle_results: List[Any],
        candidate_ids: List[str],
        strategy_name: str,
        bundle_size: int,
        model_option: Optional[RegistryModelOption],
) -> None:
    """Persist bundling outputs + metadata for later reports."""
    if not bundle_results:
        return

    df = _bundle_results_to_df(bundle_results)
    if df.empty:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"bundling_{ts}"
    base_dir = _BUNDLING_BASE_DIR  # "bundling"

    try:
        predictions_uri = save_predictions(df, base_dir, name)
    except Exception as ex:
        st.warning(f"Unable to save bundling predictions: {ex}")
        return

    meta = {
        "module": "compare_and_bundling",
        "mode": "bundling",
        "base_dir": base_dir,
        "artifact_name": name,
        "timestamp": ts,
        "strategy_name": strategy_name,
        "bundle_size": int(bundle_size),
        "candidate_title_ids": candidate_ids,
        "bundles_count": len(bundle_results),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "predictions_uri": predictions_uri,
        "model": _model_meta_dict(model_option),
    }

    try:
        meta_uri = save_prediction_metadata(meta, base_dir, name)
    except Exception as ex:
        st.warning(f"Unable to save bundling metadata: {ex}")
        meta_uri = None

    st.session_state["bundling_last_predictions_uri"] = predictions_uri
    st.session_state["bundling_last_metadata_uri"] = meta_uri


# ---------- UI helpers ------------------------------------------------------


def _render_compare_results(predictions_df: pd.DataFrame) -> None:
    """
    Comparison results section.
    NOTE: No nested expanders here – only plain elements.
    """
    if predictions_df.empty:
        st.info("Run a comparison to see results here.")
        return

    st.subheader("Comparison Summary")

    compare_df = build_compare_table(predictions_df)
    st.dataframe(
        compare_df.set_index("Metric"),
        use_container_width=True,
    )

    show_raw = st.checkbox("Show raw prediction details")
    if show_raw:
        st.dataframe(predictions_df, use_container_width=True)


def _render_bundle_results(bundle_results: List[Any]) -> None:
    """
    Bundle results section.
    Only uses containers/columns – no inner expanders.
    """
    if not bundle_results:
        st.info("Generate bundle suggestions to see results here.")
        return

    st.subheader("Suggested Bundles")

    for i, bundle in enumerate(bundle_results, start=1):
        with st.container():
            st.markdown(f"### Bundle {i}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predicted Price", f"{bundle.bundle_price_raw:,.0f}")
            with col2:
                st.metric("Value Score", f"{bundle.bundle_price_score:.2f}")
            with col3:
                st.metric("Fit Score", f"{bundle.bundle_fit_score:.2f}")
            with col4:
                st.metric(
                    "Risk Score (lower is better)",
                    f"{bundle.bundle_risk_score:.2f}",
                )

            st.markdown(bundle.rationale)

            titles_df = pd.DataFrame(bundle.titles)
            display_cols = [
                c
                for c in [
                    "title_name",
                    "predicted_price",
                    "region",
                    "platform",
                    "release_year",
                    "genres",
                ]
                if c in titles_df.columns
            ]
            if display_cols:
                st.dataframe(
                    titles_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("---")


# ---------- Main render entry point (with upload + tabs) --------------------


def render_compare_and_bundling() -> None:
    """
    Render the Compare & Bundling page.

    Layout:
      - Expander above tabs: Title Input (file upload) + model selection
      - Tabs: "Compare" and "Bundling"
        - Compare tab:
            - Expander: Compare Titles (inputs)
            - Expander: Comparison Results (outputs)
        - Bundling tab:
            - Expander: Bundling Suggestions (inputs)
            - Expander: Bundle Results (outputs)

    No expander is nested inside another expander.
    """

    st.title("Compare & Bundling")

    # Session state
    if "compare_predictions_df" not in st.session_state:
        st.session_state.compare_predictions_df = pd.DataFrame()
    if "bundle_results" not in st.session_state:
        st.session_state.bundle_results = []
    if "bundling_titles_df" not in st.session_state:
        st.session_state.bundling_titles_df = None

    # ---- Title input + model selection zone (above tabs) -------------------
    with st.expander("Title Input & Model", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload titles file (CSV)",
            type=["csv"],
            help=(
                "Provide a titles file including columns like title_id/tconst, "
                "title_name, territory/region, media, platform, license_type, etc."
            ),
        )

        if uploaded_file is not None:
            df = _load_titles_file(uploaded_file)
            st.session_state.bundling_titles_df = df

        if st.session_state.bundling_titles_df is not None:
            st.success("Titles file loaded.")
            st.dataframe(
                st.session_state.bundling_titles_df.head(),
                use_container_width=True,
            )
        else:
            st.info(
                "Upload a titles file to enable selection in Compare and Bundling tabs."
            )

        model_option = _select_registry_model()

    titles_df: Optional[pd.DataFrame] = st.session_state.bundling_titles_df
    title_options: Dict[str, str] = (
        _build_title_options_from_df(titles_df) if titles_df is not None else {}
    )

    tab_compare, tab_bundling = st.tabs(["Compare", "Bundling"])

    # --------- Compare tab --------------------------------------------------
    with tab_compare:
        # Inputs
        with st.expander("Compare Titles", expanded=True):
            if not title_options:
                st.warning(
                    "No titles available. Please upload a valid titles file above."
                )
                selected_labels: List[str] = []
                selected_ids: List[str] = []
                run_compare = False
            else:
                title_labels = list(title_options.values())
                title_ids = list(title_options.keys())

                selected_labels = st.multiselect(
                    "Select titles to compare",
                    options=title_labels,
                )

                selected_ids = [
                    title_ids[title_labels.index(lbl)]
                    for lbl in selected_labels
                    if lbl in title_labels
                ]

                col_compare_button, _ = st.columns([1, 3])
                with col_compare_button:
                    run_compare = st.button("Predict & Compare", type="primary")

                if run_compare and not selected_ids:
                    st.warning("Please select at least one title to compare.")
                    run_compare = False

                if run_compare and selected_ids:
                    predictions_df = _fetch_predictions_for_titles(
                        selected_title_ids=selected_ids,
                        titles_df=titles_df,
                        model_option=model_option,
                    )
                    st.session_state.compare_predictions_df = predictions_df
                    _persist_compare_run(
                        predictions_df=predictions_df,
                        selected_ids=selected_ids,
                        selected_labels=selected_labels,
                        model_option=model_option,
                    )

        # Results
        with st.expander(
                "Comparison Results",
                expanded=bool(not st.session_state.compare_predictions_df.empty),
        ):
            _render_compare_results(st.session_state.compare_predictions_df)

    # --------- Bundling tab -------------------------------------------------
    with tab_bundling:
        # Inputs
        with st.expander("Bundling Suggestions", expanded=True):
            if not title_options:
                st.warning(
                    "No titles available. Please upload a valid titles file above."
                )
                run_bundle = False
                candidate_ids: List[str] = []
                strategy_name = ""
                bundle_size = 0
            else:
                strategies = get_available_strategies()
                strategy_display_to_name = {s.display_name: s.name for s in strategies}
                strategy_label = st.selectbox(
                    "Bundle optimization strategy",
                    options=list(strategy_display_to_name.keys()),
                    index=1,  # default to "Balanced"
                )
                strategy_name = strategy_display_to_name[strategy_label]

                bundle_size = st.number_input(
                    "Number of titles per bundle",
                    min_value=2,
                    max_value=10,
                    value=3,
                    step=1,
                )

                title_labels = list(title_options.values())
                title_ids = list(title_options.keys())

                selected_bundle_labels = st.multiselect(
                    "Candidate titles for bundling (optional; leave empty to use all available)",
                    options=title_labels,
                )

                if selected_bundle_labels:
                    candidate_ids = [
                        title_ids[title_labels.index(lbl)]
                        for lbl in selected_bundle_labels
                        if lbl in title_labels
                    ]
                else:
                    candidate_ids = title_ids

                col_bundle_button, _ = st.columns([1, 3])
                with col_bundle_button:
                    run_bundle = st.button("Generate Bundles", type="primary")

                if run_bundle and len(candidate_ids) < bundle_size:
                    st.warning(
                        "Not enough candidate titles to form a bundle with the selected size."
                    )
                    run_bundle = False

                if run_bundle:
                    candidate_df = _fetch_predictions_for_titles(
                        selected_title_ids=candidate_ids,
                        titles_df=titles_df,
                        model_option=model_option,
                    )
                    bundle_results = generate_candidate_bundles(
                        candidate_df=candidate_df,
                        bundle_size=int(bundle_size),
                        strategy_name=strategy_name,
                        max_anchors=10,
                        max_bundles=5,
                    )
                    st.session_state.bundle_results = bundle_results
                    _persist_bundling_run(
                        bundle_results=bundle_results,
                        candidate_ids=candidate_ids,
                        strategy_name=strategy_name,
                        bundle_size=int(bundle_size),
                        model_option=model_option,
                    )

        # Results
        with st.expander(
                "Bundle Results",
                expanded=bool(st.session_state.bundle_results),
        ):
            _render_bundle_results(st.session_state.bundle_results)
