import os
from pathlib import Path

import streamlit as st

from src.ui.common import end_tab_scroll, begin_tab_scroll, section_panel
from src.utils.log_utils import get_logger
from src.validator.schema_validator import validate_df

LOGGER = get_logger("ui_display_data")


def render_display_section():
    st.header("Display Data")
    LOGGER.info("Rendering Display Data panel..")
    df = st.session_state.df
    label = os.path.basename(Path(st.session_state.last_feature_master_path))
    st.caption(f"Using: {label} — shape={df.shape}")
    with section_panel("Display Data", expanded=True):
        tabs = st.tabs(["Table (Preview)", "Schema"])

        with tabs[0]:
            begin_tab_scroll()
            preview_rows = 100 if len(df) > 100 else len(df)
            st.caption(f"Preview: showing first {preview_rows} rows (of {len(df)})")
            st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True)
            end_tab_scroll()

        with tabs[1]:
            begin_tab_scroll()
            validated_df, report = validate_df(df)
            if report["status"] == "passed":
                st.success("Schema validation passed.")
                st.json({
                    "rows": len(validated_df),
                    "cols": validated_df.shape[1],
                    "dtypes": validated_df.dtypes.astype(str).to_dict()
                })
                st.session_state.df = validated_df  # keep coerced frame
            else:
                st.warning("Schema validation failed; details below.")
                st.json(report["errors"])
            end_tab_scroll()

    LOGGER.info("Display Data panel rendered")
