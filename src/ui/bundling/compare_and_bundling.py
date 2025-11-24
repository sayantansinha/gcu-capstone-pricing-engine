from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

from src.services.compare_and_bundling.bundling_service import (
    get_available_strategies,
    generate_candidate_bundles,
)
from src.services.compare_and_bundling.comparison_service import build_compare_table


# ---------- Integration hooks to your existing services ---------------------
def _fetch_predictions_for_titles(selected_title_ids: List[str]) -> pd.DataFrame:
    """
    Wire this to your existing price prediction service.

    You will likely want to:
      - Look up row metadata from st.session_state.bundling_titles_df
      - Call your price prediction service per title_id (or in batch)
      - Return a DataFrame including at least:

        - title_id
        - title_name
        - predicted_price
        - region / territory
        - platform
        - release_year
        - genres
        - (optional) popularity
        - (optional) risk_proxy

    For now this is a stub so the UI does not crash.
    """
    if not selected_title_ids:
        return pd.DataFrame()

    data = []
    for i, tid in enumerate(selected_title_ids):
        data.append(
            {
                "title_id": tid,
                "title_name": f"Title {tid}",
                "predicted_price": 100_000 + i * 10_000,
                "region": "US",
                "platform": "SVOD",
                "release_year": 2020 + (i % 3),
                "genres": "Drama|Thriller" if i % 2 == 0 else "Comedy",
                "popularity": 0.5 + 0.05 * i,
                "risk_proxy": 0.4 + 0.02 * i,
            }
        )
    return pd.DataFrame(data)


# ---------- Title input & options helpers ----------------------------------


def _load_titles_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load the uploaded titles file into a DataFrame.
    Currently supports CSV; you can extend this to parquet etc. later.
    """
    if uploaded_file is None:
        return None

    # Basic support – assume CSV for now
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
      - Expander above tabs: Title Input (file upload)
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

    # ---- Title input zone (above tabs) ------------------------------------
    with st.expander("Title Input", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload titles file (CSV)",
            type=["csv"],
            help="Provide a titles file including columns like title_id/tconst, "
                 "title_name, territory/region, media, platform, license_type, etc.",
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

                if run_compare and selected_ids:
                    predictions_df = _fetch_predictions_for_titles(selected_ids)
                    st.session_state.compare_predictions_df = predictions_df

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
                elif run_bundle:
                    candidate_df = _fetch_predictions_for_titles(candidate_ids)
                    bundle_results = generate_candidate_bundles(
                        candidate_df=candidate_df,
                        bundle_size=bundle_size,
                        strategy_name=strategy_name,
                        max_anchors=10,
                        max_bundles=5,
                    )
                    st.session_state.bundle_results = bundle_results

        # Results
        with st.expander(
                "Bundle Results",
                expanded=bool(st.session_state.bundle_results),
        ):
            _render_bundle_results(st.session_state.bundle_results)
