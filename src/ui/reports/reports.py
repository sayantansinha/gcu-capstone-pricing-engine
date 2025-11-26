# src/ui/reports/reports.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
import io
import zipfile

import pandas as pd
import streamlit as st

from src.services.reports.reports_service import (
    build_portfolio_overview,
    build_territory_intelligence,
    build_platform_strategy,
    build_bundle_value_report,
    build_executive_summary,
)


# ---------------------------------------------------------------------
# Helpers – get REAL data from session_state and flatten bundles
# ---------------------------------------------------------------------
def _get_pricing_df_from_session() -> Optional[pd.DataFrame]:
    """
    Get the pricing/prediction dataframe produced by the Compare part
    of the Compare & Bundling module.

    Compare & Bundling stores this as `compare_predictions_df`
    in session_state. If it's missing or empty, we return None.
    """
    df = st.session_state.get("compare_predictions_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _flatten_bundle_results(bundle_results: List[Any]) -> pd.DataFrame:
    """
    Flatten BundleResult objects (from Compare & Bundling) into a row-level DataFrame.

    This mirrors `_bundle_results_to_df` in compare_and_bundling.py but is
    defined here to avoid UI module cross-imports.

    Columns:
      - bundle_id, strategy_name
      - bundle_price_raw
      - bundle_value_score
      - bundle_fit_score
      - bundle_diversity_score
      - bundle_risk_score
      - title_id, title_name, predicted_price, region, platform,
        release_year, genres
    """
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
    """
    Build a bundling DataFrame from the in-memory bundle_results produced
    by the Bundling tab in Compare & Bundling.

    If there are no bundle_results, or if flattening yields an empty
    DataFrame, we return None.
    """
    bundle_results = st.session_state.get("bundle_results")
    if not bundle_results:
        return None

    df = _flatten_bundle_results(bundle_results)
    if df.empty:
        return None
    return df


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------
def render_reports() -> None:
    """
    Business-facing Pricing & Licensing Reports page.

    Uses REAL data from:

      - Compare & Bundling:
          * st.session_state.compare_predictions_df  → pricing_df
          * st.session_state.bundle_results          → bundling_df (flattened)

    We intentionally do NOT pull from the Price Predictor audits yet,
    because those are persisted via audit helpers and not exposed as
    DataFrames in session_state. This avoids guessing file locations
    or formats.
    """
    st.title("Reports")

    pricing_df = _get_pricing_df_from_session()
    if pricing_df is None:
        st.info(
            "Pricing reports require predictions from the **Compare** tab. "
            "Please run a comparison in the Compare & Bundling screen first."
        )
        return

    bundling_df = _get_bundling_df_from_session()

    # Build all aggregate views once so we can use them in tabs and downloads
    overview = build_portfolio_overview(pricing_df, top_n=20)
    ti = build_territory_intelligence(pricing_df)
    ps = build_platform_strategy(pricing_df)
    bundle_report = build_bundle_value_report(bundling_df) if bundling_df is not None else None
    es = build_executive_summary(pricing_df)

    # -----------------------------------------------------------------
    # Global download section
    # -----------------------------------------------------------------
    st.markdown("### Downloads")

    # Download full predictions as CSV
    pricing_csv = pricing_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download full predictions (CSV)",
        data=pricing_csv,
        file_name="pricing_predictions_full.csv",
        mime="text/csv",
        key="download_pricing_full_csv",
    )

    # Download all report artifacts as a ZIP of CSVs
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Core dataset
        zf.writestr("pricing_predictions_full.csv", pricing_df.to_csv(index=False))

        # Portfolio
        if not overview.by_title.empty:
            zf.writestr("portfolio_by_title.csv", overview.by_title.to_csv(index=False))
        if overview.by_genre is not None and not overview.by_genre.empty:
            zf.writestr("portfolio_by_genre.csv", overview.by_genre.to_csv(index=False))

        # Territory
        if not ti.by_territory.empty:
            zf.writestr("territory_summary.csv", ti.by_territory.to_csv(index=False))

        # Platform
        if not ps.by_platform.empty:
            zf.writestr("platform_summary.csv", ps.by_platform.to_csv(index=False))

        # Bundles
        if bundle_report is not None and not bundle_report.bundles.empty:
            zf.writestr("bundle_summary.csv", bundle_report.bundles.to_csv(index=False))

        # Executive views
        if not es.top_titles.empty:
            zf.writestr("exec_top_titles.csv", es.top_titles.to_csv(index=False))
        if not es.top_territories.empty:
            zf.writestr("exec_top_territories.csv", es.top_territories.to_csv(index=False))
        if not es.top_platforms.empty:
            zf.writestr("exec_top_platforms.csv", es.top_platforms.to_csv(index=False))

    zip_buf.seek(0)
    st.download_button(
        label="⬇️ Download all report data (ZIP)",
        data=zip_buf,
        file_name="pricing_reports_all.csv.zip",
        mime="application/zip",
        key="download_all_reports_zip",
    )

    st.markdown("---")

    tab_portfolio, tab_territory, tab_platform, tab_bundle, tab_exec = st.tabs(
        [
            "Portfolio Overview",
            "Territory Intelligence",
            "Platform Strategy",
            "Bundle Optimization",
            "Executive Summary",
        ]
    )

    # -----------------------------------------------------------------
    # 1. Portfolio Pricing Overview
    # -----------------------------------------------------------------
    with tab_portfolio:
        st.subheader("Portfolio Pricing Overview")

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

        st.markdown("### Top Titles by Predicted Value")
        if not overview.by_title.empty:
            st.dataframe(overview.by_title, use_container_width=True)

            # Per-tab download
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
                "Make sure your Compare predictions include `title_id` or `title_name`."
            )

        # Will be implemented once genre is integrated into the pipeline
        # st.markdown("### Genre-Level Value (if available)")
        # if overview.by_genre is not None and not overview.by_genre.empty:
        #     st.dataframe(overview.by_genre, use_container_width=True)
        #     genre_csv = overview.by_genre.to_csv(index=False).encode("utf-8")
        #     st.download_button(
        #         label="⬇️ Download portfolio by genre (CSV)",
        #         data=genre_csv,
        #         file_name="portfolio_by_genre.csv",
        #         mime="text/csv",
        #         key="download_portfolio_by_genre_csv",
        #     )
        # else:
        #     st.info("No `genre`/`genres` column detected; genre-level summary not available.")

    # -----------------------------------------------------------------
    # 2. Territory Intelligence
    # -----------------------------------------------------------------
    with tab_territory:
        st.subheader("Territory Intelligence Report")

        if ti.by_territory.empty:
            st.info(
                "This report needs either a `territory` or `region` column in the "
                "Compare predictions. Currently neither was found."
            )
        else:
            st.markdown("### Territory Summary")
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
                st.markdown("### Average Predicted Price by Territory")
                chart_df = ti.by_territory.set_index("territory")["avg_predicted"]
                st.bar_chart(chart_df)

    # -----------------------------------------------------------------
    # 3. Platform Strategy
    # -----------------------------------------------------------------
    with tab_platform:
        st.subheader("Platform Strategy Report")

        if ps.by_platform.empty:
            st.info(
                "This report needs a `platform` column in the Compare predictions. "
                "Currently no platform information was found."
            )
        else:
            st.markdown("### Platform Summary")
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
                st.markdown("### Average Predicted Price by Platform")
                chart_df = ps.by_platform.set_index("platform")["avg_predicted"]
                st.bar_chart(chart_df)

    # -----------------------------------------------------------------
    # 4. Bundle Value Optimization
    # -----------------------------------------------------------------
    with tab_bundle:
        st.subheader("Bundle Value Optimization Report")

        if bundling_df is None:
            st.info(
                "No bundling results found in this session. "
                "Run the **Bundling** tab in Compare & Bundling to populate this report."
            )
        else:
            if bundle_report is None or bundle_report.bundles.empty:
                st.info(
                    "Bundling results are present but do not contain `bundle_id` and "
                    "`bundle_price_raw`. Update the bundling output schema or adjust "
                    "`build_bundle_value_report` if your schema differs."
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

                st.markdown("### Bundle-Level Summary")
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
    # 5. Executive Pricing Intelligence Summary
    # -----------------------------------------------------------------
    with tab_exec:
        st.subheader("Executive Pricing Intelligence Summary")

        st.markdown("### Top Titles by Predicted Value")
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
                "Ensure `title_id`/`title_name` and `predicted_price` are present in Compare predictions."
            )

        st.markdown("### Top Territories by Total Predicted Value")
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
                "Add `territory` or `region` to the Compare predictions to enable this view."
            )

        st.markdown("### Top Platforms by Total Predicted Value")
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
                "Add `platform` to the Compare predictions to enable this view."
            )
