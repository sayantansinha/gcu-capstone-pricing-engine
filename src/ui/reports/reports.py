from pathlib import Path
import io
import zipfile

import streamlit as st

from src.config.env_loader import SETTINGS
from src.utils.data_io_utils import (
    list_report_runs,
    list_reports_for_run,
    latest_report_for_run,
    load_report_for_download,
)


def render_reports() -> None:
    """
    Top-level Reports page (NOT the pipeline 'reports' step).

    - Lists all runs that have saved PDFs (under REPORTS_DIR or REPORTS_BUCKET).
    - Lets user download reports per run.
    - Provides a simple Export Center to download all reports as a zip.
    """
    st.title("Reports Center")

    # 1) Discover runs that actually have reports
    run_ids = list_report_runs()

    if not run_ids:
        st.info("No reports found yet. Run the pipeline with reporting enabled to generate reports.")
        return

    # -----------------------------
    # A. Run Summary Reports
    # -----------------------------
    with st.expander("Run Summary Reports", expanded=True):
        selected_run = st.selectbox(
            "Select a pipeline run",
            options=run_ids,
            index=0,
            key="reports_run_select",
        )

        if selected_run:
            st.markdown(f"**Selected run:** `{selected_run}`")

            # Latest report (based on your naming convention, via latest_report_for_run)
            latest_ref = latest_report_for_run(selected_run)
            if latest_ref:
                try:
                    pdf_bytes = load_report_for_download(latest_ref)
                    latest_name = (
                        Path(latest_ref).name
                        if not str(latest_ref).startswith("s3://")
                        else str(latest_ref).split("/")[-1]
                    )
                    st.download_button(
                        label=f"⬇️ Download latest report ({latest_name})",
                        data=pdf_bytes,
                        file_name=latest_name,
                        mime="application/pdf",
                        key="download_latest_report",
                    )
                    st.caption(f"Source: `{latest_ref}`")
                except Exception as ex:
                    st.error(f"Could not load latest report for run {selected_run}: {ex}")
            else:
                st.warning("No report PDF found for this run (yet).")

            st.markdown("---")

            # List all report PDFs for this run (e.g., multiple types per run)
            all_reports = list_reports_for_run(selected_run)
            if all_reports:
                st.subheader("All reports for this run")
                for ref in all_reports:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(str(ref), language="text")
                    with col2:
                        try:
                            pdf_bytes = load_report_for_download(ref)
                            fname = (
                                Path(ref).name
                                if not str(ref).startswith("s3://")
                                else str(ref).split("/")[-1]
                            )
                            st.download_button(
                                label="Download",
                                data=pdf_bytes,
                                file_name=fname,
                                mime="application/pdf",
                                key=f"download_{selected_run}_{fname}",
                            )
                        except Exception as ex:
                            st.error(f"Error loading {ref}: {ex}")
            else:
                st.info("This run does not yet have any PDF reports stored.")

    # -----------------------------
    # B. Export Center
    # -----------------------------
    with st.expander("Export Center"):
        st.write(
            "Download a consolidated zip containing **all** report PDFs for all runs "
            "currently available in the reports storage."
        )

        if st.button("Prepare ZIP of all report PDFs", key="prepare_all_reports_zip"):
            buf = io.BytesIO()

            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for run_id in run_ids:
                    report_refs = list_reports_for_run(run_id)
                    for ref in report_refs:
                        try:
                            pdf_bytes = load_report_for_download(ref)

                            # Derive a reasonable archive path inside the zip
                            if str(ref).startswith("s3://"):
                                # For S3, drop the scheme and bucket name, keep the key
                                parts = str(ref).split("/", 3)
                                arcname = parts[3] if len(parts) >= 4 else Path(ref).name
                            else:
                                # For local, make path relative to REPORTS_DIR when possible
                                p = Path(ref)
                                reports_root = Path(SETTINGS.REPORTS_DIR)
                                try:
                                    arcname = str(p.relative_to(reports_root))
                                except ValueError:
                                    arcname = p.name

                            zf.writestr(arcname, pdf_bytes)
                        except Exception as ex:
                            # Don't break the whole export; just skip problematic files
                            st.warning(f"Skipping {ref} due to error: {ex}")

            buf.seek(0)
            st.download_button(
                label="⬇️ Download all reports (ZIP)",
                data=buf,
                file_name="all_reports.zip",
                mime="application/zip",
                key="download_all_reports_zip",
            )
