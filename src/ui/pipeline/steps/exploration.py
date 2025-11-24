import os

import streamlit as st

from src.services.pipeliine.source_data.analytics.eda_service import eda_summary
from src.services.pipeliine.source_data.analytics.visualization_service import plot_hist, plot_box, plot_bar, plot_scatter, \
    plot_datetime_counts, \
    plot_time_of_day_hist, plot_month_box
from src.ui.common import end_tab_scroll, begin_tab_scroll, section_panel, load_active_feature_master_from_session
from src.utils.data_io_utils import load_figure
from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_exploration")


def _display_figure(path_or_uri: str, *, context: str) -> None:
    """
    Display a figure from LOCAL or S3 and handle errors gracefully.
    """
    caption = os.path.basename(str(path_or_uri))
    try:
        img = load_figure(path_or_uri)
        st.image(img, caption=caption)
    except Exception as e:
        LOGGER.exception(
            "Error displaying visualization (%s): %s", context, path_or_uri, exc_info=e
        )
        st.error(
            f"An error occurred while rendering the {context} visualization. "
            "Please check the application logs for details."
        )


def _viz_numeric_cat(df, run_id):
    try:
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in all_cols if c not in num_cols]

        chart = st.selectbox("Chart type", ["Histogram", "Boxplot", "Bar", "Scatter"], key="viz_chart")
        if chart == "Histogram":
            if not num_cols:
                st.info("No numeric columns.")
                return
            c = st.selectbox("Column", num_cols, key="hist_col")
            bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
            if st.button("Plot histogram"):
                p = plot_hist(df, c, run_id, bins=bins)
                _display_figure(p, context="histogram")
        elif chart == "Boxplot":
            if not num_cols:
                st.info("No numeric columns.")
                return
            c = st.selectbox("Column", num_cols, key="box_col")
            if st.button("Plot boxplot"):
                p = plot_box(df, c, run_id)
                _display_figure(p, context="boxplot")
        elif chart == "Bar":
            cat = st.selectbox("Category", cat_cols or all_cols, key="bar_cat")
            val = st.selectbox("Aggregate (optional numeric)", [None] + num_cols, key="bar_val")
            if st.button("Plot bar"):
                p = plot_bar(df, cat, val, run_id)
                _display_figure(p, context="bar chart")
        else:
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns.")
                return
            x = st.selectbox("X", num_cols, key="scatter_x")
            y = st.selectbox("Y", num_cols, key="scatter_y")
            if st.button("Plot scatter"):
                p = plot_scatter(df, x, y, run_id)
                _display_figure(p, context="scatter")
    except Exception as e:
        LOGGER.exception(
            "Error in Visualization – Numeric/Categorical tab for run_id=%s",
            run_id,
            exc_info=e,
        )
        st.error(
            "An unexpected error occurred while generating numeric/categorical visualizations. "
            "Please check the logs for details."
        )


def _viz_datetime(df, run_id):
    try:
        dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
        if not dt_cols:
            st.info("No datetime columns detected.")
            return

        choice = st.selectbox("Datetime chart", ["Counts over time", "Time-of-day", "Value by month"], key="dt_choice")
        if choice == "Counts over time":
            dc = st.selectbox("Datetime column", dt_cols, key="dt_counts_col")
            freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=2, key="dt_counts_freq")
            if st.button("Plot counts over time"):
                p = plot_datetime_counts(df, dc, run_id, freq=freq)
                _display_figure(p, context="daily counts")
        elif choice == "Time-of-day":
            dc = st.selectbox("Datetime column", dt_cols, key="dt_hour_col")
            if st.button("Plot hour histogram"):
                p = plot_time_of_day_hist(df, dc, run_id)
                _display_figure(p, context="hour-of-day histogram")
        else:
            dc = st.selectbox("Datetime column", dt_cols, key="dt_month_col")
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                st.info("Need a numeric value column for boxplot.")
                return
            vc = st.selectbox("Numeric value", num_cols, key="dt_month_val")
            if st.button("Plot month boxplot"):
                p = plot_month_box(df, dc, vc, run_id)
                _display_figure(p, context="month boxplot")
    except Exception as e:
        LOGGER.exception(
            "Error in Visualization – Datetime tab for run_id=%s",
            run_id,
            exc_info=e,
        )
        st.error(
            "An unexpected error occurred while generating datetime visualizations. "
            "Please check the logs for details."
        )


def render_exploration_section():
    run_id = st.session_state.run_id
    LOGGER.info(f"Rendering Exploration (EDA) panel.....run_id: {run_id}")
    st.header("Exploration (EDA)")
    df, label = load_active_feature_master_from_session()
    if df is None:
        st.warning("No feature master found. Build it in Data Staging.")
        return
    st.caption(f"Using: {label} — shape={df.shape}")
    df = st.session_state.df

    with section_panel("Exploration (EDA)", expanded=True):
        tabs = st.tabs(["Describe", "Numeric Correlations", "Visualization"])

        with tabs[0]:
            begin_tab_scroll()
            profile = eda_summary(df)
            st.json(profile.get("describe", {}))
            end_tab_scroll()

        with tabs[1]:
            begin_tab_scroll()
            profile = eda_summary(df)
            st.json(profile.get("corr_numeric", {}))
            end_tab_scroll()

        with tabs[2]:
            subtabs = st.tabs(["Numeric/Categorical", "Datetime"])
            with subtabs[0]:
                begin_tab_scroll()
                _viz_numeric_cat(df, run_id)
                end_tab_scroll()
            with subtabs[1]:
                begin_tab_scroll()
                _viz_datetime(df, run_id)
                end_tab_scroll()

    LOGGER.info(f"Exploration (EDA) panel rendered.....run_id: {run_id}")
