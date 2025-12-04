import _io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.config.env_loader import SETTINGS
from src.utils.log_utils import get_logger
from src.utils.s3_utils import (
    ensure_bucket,
    list_bucket_objects,
    load_bucket_object,
    write_dataframe_parquet,
    write_bucket_object,
    formulate_s3_uri,
)

LOGGER = get_logger("data_io_utils")


@dataclass
class RunInfo:
    run_id: str
    has_raw: bool = False
    has_feature_master: bool = False
    has_feature_master_cleaned: bool = False
    has_model: bool = False
    has_registered_model: bool = False


# -----------------------------
# Helpers: resolve where to write/read
# -----------------------------
def is_s3() -> bool:
    return SETTINGS.IO_BACKEND == "S3"


def _ensure_buckets():
    # idempotent
    for b in filter(None, [
        SETTINGS.RAW_BUCKET,
        SETTINGS.PROCESSED_BUCKET,
        SETTINGS.PROFILES_BUCKET,
        SETTINGS.FIGURES_BUCKET,
        SETTINGS.MODELS_BUCKET,
        SETTINGS.REPORTS_BUCKET,
        SETTINGS.PREDICTIONS_BUCKET
    ]):
        ensure_bucket(b)


# -----------------------------
# Raw data
# -----------------------------
def save_raw(df: pd.DataFrame, base_dir: str, name: str) -> str:
    """
    Save raw dataset either to local or S3.
    base_dir acts as a subdirectory (LOCAL) or a prefix (S3).
    """
    if is_s3():
        _ensure_buckets()
        key = f"{base_dir.strip('/')}/{name}.parquet"
        write_dataframe_parquet(df, SETTINGS.RAW_BUCKET, key, index=False)
        uri = formulate_s3_uri(SETTINGS.RAW_BUCKET, key)
        LOGGER.info(f"Saved raw (S3) → {uri}")
        return uri
    else:
        path = os.path.join(Path(SETTINGS.RAW_DIR) / base_dir, f"{name}.parquet")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        LOGGER.info(f"Saved raw (LOCAL) → {path}")
        return path


def load_raw(path_or_name: str, base_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw dataset. If S3, pass name without suffix + base_dir; if LOCAL, pass a full path.
    """
    if is_s3():
        name = path_or_name if path_or_name.endswith(".parquet") else f"{path_or_name}.parquet"
        key = f"{base_dir.strip('/')}/{name}" if base_dir else name
        LOGGER.info(f"Loading raw (S3) ← s3://{SETTINGS.RAW_BUCKET}/{key}")
        return load_bucket_object(SETTINGS.RAW_BUCKET, key)
    else:
        path = path_or_name
        LOGGER.info(f"Loading raw (LOCAL) ← {path}")
        return pd.read_parquet(path)


def list_raw_files(base_dir: str) -> List[str]:
    if is_s3():
        prefix = f"{base_dir.strip('/')}/"
        keys = [k for k in list_bucket_objects(SETTINGS.RAW_BUCKET, prefix) if k.endswith(".parquet")]
        keys.sort()
        LOGGER.info(f"Listing raw (S3) {SETTINGS.RAW_BUCKET}/{prefix} => {keys}")
        return keys
    else:
        raw = Path(SETTINGS.RAW_DIR) / base_dir
        if not os.path.isdir(raw):
            return []
        files = [str(raw / f) for f in os.listdir(raw) if f.endswith(".parquet")]
        files.sort()
        LOGGER.info(f"Listing raw (LOCAL) {raw} => {files}")
        return files


def _read_parquet_head(path_or_file, n_rows: int = 50000, columns=None) -> pd.DataFrame:
    """
    Read only the first n_rows rows from a Parquet file efficiently.
    Accepts a local path, S3 URL, or a file-like object.
    """
    pf = pq.ParquetFile(path_or_file)

    tables = []
    rows_read = 0

    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=columns)
        tables.append(table)
        rows_read += table.num_rows

        if rows_read >= n_rows:
            break

    if not tables:
        return pd.DataFrame()

    combined = pa.concat_tables(tables)
    combined = combined.slice(0, min(n_rows, combined.num_rows))
    return combined.to_pandas()


def _read_parquet_head_from_url(url: str, n_rows: int = 50000, columns=None) -> pd.DataFrame:
    with fsspec.open(url, "rb") as f:
        return _read_parquet_head(f, n_rows, columns)


def save_from_url(url: str, base_dir: str, cap_n_rows: int = 50000) -> str:
    """
    Read from a remote URL (parquet, tsv/tsv.gz, or generic CSV),
    optionally sampling only the first `sample_rows` rows, and save as raw parquet.
    """
    try:
        LOGGER.info(f"Reading data from URL: {url}")

        # Strip query params for suffix detection and basename
        parsed = urlparse(url)
        path_only = parsed.path  # e.g. '/datasets/title.basics.tsv.gz'

        lower_path = path_only.lower()
        base_name = Path(path_only).stem.replace(".tsv", "").replace(".gz", "") or "remote"

        if lower_path.endswith((".parquet", ".pq")):
            df = _read_parquet_head_from_url(url, n_rows=cap_n_rows)
        elif lower_path.endswith((".tsv.gz", ".tsv")):
            LOGGER.info("Parsing TSV data from URL")
            df = pd.read_csv(
                url,
                sep="\t",
                compression="infer",
                na_values="\\N",
                low_memory=False,
                nrows=cap_n_rows
            )
        else:
            LOGGER.info("Parsing text data from URL")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(
                _io.StringIO(r.text),
                low_memory=False,
                nrows=cap_n_rows
            )

        return save_raw(df, base_dir, base_name)
    except Exception as ex:
        error_text = f"Error reading file from {url}"
        LOGGER.exception(error_text)
        raise ValueError(error_text) from ex


# -----------------------------
# Processed data
# -----------------------------
def save_processed(df: pd.DataFrame, base_dir: str, name: str) -> str:
    if is_s3():
        _ensure_buckets()
        key = f"{base_dir.strip('/')}/{name}.parquet"
        write_dataframe_parquet(df, SETTINGS.PROCESSED_BUCKET, key, index=False)
        uri = formulate_s3_uri(SETTINGS.PROCESSED_BUCKET, key)
        LOGGER.info(f"Saved processed (S3) → {uri}")
        return uri
    else:
        path = os.path.join(Path(SETTINGS.PROCESSED_DIR) / base_dir, f"{name}.parquet")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        LOGGER.info(f"Saved processed (LOCAL) → {path}")
        return path


def load_processed(name: str, base_dir: str = None) -> pd.DataFrame:
    if is_s3():
        name = name if name.endswith(".parquet") else f"{name}.parquet"
        key = f"{base_dir.strip('/')}/{name}" if base_dir else name
        LOGGER.info(f"Loading processed (S3) ← s3://{SETTINGS.PROCESSED_BUCKET}/{key}")
        return load_bucket_object(SETTINGS.PROCESSED_BUCKET, key)
    else:
        name = name if name.endswith(".parquet") else f"{name}.parquet"
        path = os.path.join(Path(SETTINGS.PROCESSED_DIR) / base_dir, name)
        LOGGER.info(f"Loading processed (LOCAL) ← {path}")
        return pd.read_parquet(path)


# -----------------------------
# Figures
# -----------------------------
def save_figure(fig, base_dir: str, name: str) -> str:
    """
    Save a matplotlib figure either locally or to S3.

    Returns:
        LOCAL → filesystem path as str
        S3    → s3://bucket/key URI
    """
    # normalise name so we don't end up with .png.png
    if name.lower().endswith(".png"):
        base_name = name
    else:
        base_name = f"{name}.png"

    if is_s3():
        _ensure_buckets()

        buf = BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)

        key = f"{base_dir.strip('/')}/{base_name}"
        write_bucket_object(
            SETTINGS.FIGURES_BUCKET,
            key,
            buf.read(),
            content_type="image/png",
        )
        uri = formulate_s3_uri(SETTINGS.FIGURES_BUCKET, key)
        LOGGER.info(f"Saved figure (S3) → {uri}")
        return uri
    else:
        path = Path(SETTINGS.FIGURES_DIR) / base_dir / base_name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        LOGGER.info(f"Saved figure (LOCAL) → {path}")
        return str(path)


def load_figure(path_or_uri: str):
    """
    Load a figure for display in Streamlit.

    LOCAL:
        returns a filesystem path (Streamlit will open it)
    S3:
        returns raw bytes that st.image can render
    """
    if isinstance(path_or_uri, str) and path_or_uri.startswith("s3://"):
        try:
            with fsspec.open(path_or_uri, "rb") as f:
                data = f.read()
            return data
        except Exception as ex:
            LOGGER.exception(f"Error loading figure from {path_or_uri}: {ex}")
            raise
    # for local paths, just return the string
    return path_or_uri


# -----------------------------
# Profiles / validation summaries
# -----------------------------
def save_profile(profile_obj: dict, base_dir: str, name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{name}_{timestamp}.json"
    if is_s3():
        _ensure_buckets()
        key = f"{base_dir.strip('/')}/{filename}"
        write_bucket_object(
            SETTINGS.PROFILES_BUCKET,
            key,
            json.dumps(profile_obj, indent=2),
            content_type="application/json"
        )
        uri = formulate_s3_uri(SETTINGS.PROFILES_BUCKET, key)
        LOGGER.info(f"Saved profile (S3) → {uri}")
        return uri
    else:
        path = os.path.join(Path(SETTINGS.PROFILES_DIR) / base_dir, filename)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile_obj, f, indent=2)
        LOGGER.info(f"Saved profile (LOCAL) → {path}")
        return path


# -----------------------------
# Latest file under (LOCAL or S3)
# -----------------------------
def latest_file_under_directory(
        prefix: str,
        under_dir: Path | None = None,
        suffix: str = ".parquet",
        exclusion: str = None
) -> Optional[str]:
    if is_s3():
        # Default to processed bucket / root if no base provided
        base_prefix = "" if under_dir is None else str(under_dir).strip("/")
        search = f"{base_prefix}/{prefix}".strip("/")
        keys = [k for k in list_bucket_objects(SETTINGS.PROCESSED_BUCKET, search)
                if k.startswith(search) and k.endswith(suffix) and (exclusion is None or exclusion not in k)]
        if not keys:
            LOGGER.warning(f"No keys under s3://{SETTINGS.PROCESSED_BUCKET}/{search}")
            return None
        keys.sort(reverse=True)
        latest = keys[0]
        return formulate_s3_uri(SETTINGS.PROCESSED_BUCKET, latest)
    else:
        if under_dir is None:
            LOGGER.warning("under_dir is required for LOCAL backend")
            return None
        if not under_dir.exists():
            LOGGER.warning(f"Directory {under_dir.name} doesn't exist")
            return None
        files = [p for p in under_dir.iterdir()
                 if p.is_file()
                 and p.name.startswith(prefix)
                 and p.suffix == suffix
                 and (exclusion is None or exclusion not in p.name)]
        if not files:
            LOGGER.warning(f"No files found under {under_dir.name} directory")
            return None
        files.sort(key=lambda p: p.name, reverse=True)
        return str(files[0])


# -----------------------------
# Reports (PDF)
# -----------------------------
def save_report_pdf(base_dir: str, name: str, pdf_bytes: bytes) -> str:
    """
    Save a PDF report either locally or to S3.

    Args:
        base_dir: run_id or logical subdirectory under reports root.
        name: filename WITHOUT extension ('.pdf' will be added).
        pdf_bytes: raw PDF bytes.

    Returns:
        LOCAL → filesystem path as str
        S3    → s3://bucket/key URI
    """
    filename = f"{name}.pdf"

    if is_s3():
        _ensure_buckets()
        bucket = getattr(SETTINGS, "REPORTS_BUCKET", None)
        if not bucket:
            LOGGER.warning("REPORTS_BUCKET not configured; save_report_pdf → no-op")
            return ""

        key = f"{base_dir.strip('/')}/{filename}"
        write_bucket_object(
            bucket,
            key,
            pdf_bytes,
            content_type="application/pdf",
        )
        uri = formulate_s3_uri(bucket, key)
        LOGGER.info("Saved report (S3) → %s", uri)
        return uri
    else:
        out_path = Path(SETTINGS.REPORTS_DIR) / base_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        LOGGER.info("Saved report (LOCAL) → %s", out_path)
        return str(out_path)


def latest_report_for_run(
        run_id: str,
        prefix: str = "report_",
        suffix: str = ".pdf",
) -> Optional[str]:
    """
    Find the latest report for a run_id.

    Returns:
        LOCAL → filesystem path as str
        S3    → s3://bucket/key URI
        None  → if nothing found
    """
    if is_s3():
        bucket = getattr(SETTINGS, "REPORTS_BUCKET", None)
        if not bucket:
            LOGGER.warning("REPORTS_BUCKET not configured; latest_report_for_run → None")
            return None

        base_prefix = run_id.strip("/")
        search = f"{base_prefix}/{prefix}".strip("/")

        keys = [
            k for k in list_bucket_objects(bucket, search)
            if k.startswith(search) and k.endswith(suffix)
        ]
        if not keys:
            LOGGER.info("No report keys under s3://%s/%s", bucket, base_prefix)
            return None

        keys.sort(reverse=True)
        latest_key = keys[0]
        return formulate_s3_uri(bucket, latest_key)
    else:
        reports_dir = Path(SETTINGS.REPORTS_DIR) / run_id
        if not reports_dir.exists():
            LOGGER.info("Reports dir for run_id %s does not exist: %s", run_id, reports_dir)
            return None

        files = [
            p for p in reports_dir.iterdir()
            if p.is_file() and p.name.startswith(prefix) and p.suffix == suffix
        ]
        if not files:
            LOGGER.info("No report files found under %s", reports_dir)
            return None

        files.sort(key=lambda p: p.name, reverse=True)
        return str(files[0])


def load_report_for_download(path_or_ref: str) -> bytes:
    """
    Load a report (PDF) into bytes suitable for st.download_button.

    Accepts:
        - Local filesystem path
        - s3://bucket/key URI
        - The slightly-mangled 's3:/bucket/key' that comes from Path("s3://...")

    Returns:
        Raw PDF bytes.
    """
    ref = str(path_or_ref)

    # Normalize the common 's3:/bucket/key' → 's3://bucket/key'
    if ref.startswith("s3:/") and not ref.startswith("s3://"):
        ref = "s3://" + ref[len("s3:/"):]

    if ref.startswith("s3://"):
        try:
            with fsspec.open(ref, "rb") as f:
                return f.read()
        except Exception as ex:
            LOGGER.exception("Error loading report from %s: %s", ref, ex)
            raise
    else:
        p = Path(ref)
        with open(p, "rb") as f:
            return f.read()


# -----------------------------
# Reports listing helpers
# -----------------------------
def list_report_runs() -> List[str]:
    """
    List run_ids (base_dir) that have at least one report saved,
    abstracting over LOCAL vs S3.
    """
    if is_s3():
        bucket = getattr(SETTINGS, "REPORTS_BUCKET", None)
        if not bucket:
            LOGGER.warning("REPORTS_BUCKET not configured; list_report_runs → []")
            return []

        # List all objects under the reports bucket and infer run_ids
        keys = list_bucket_objects(bucket, prefix="")
        run_ids: set[str] = set()
        for key in keys:
            # Only consider PDF report objects
            if not key.endswith(".pdf"):
                continue
            # run_id is first path segment
            parts = key.split("/", 1)
            if parts and parts[0]:
                run_ids.add(parts[0].strip("/"))
        return sorted(run_ids)
    else:
        reports_root = Path(SETTINGS.REPORTS_DIR)
        if not reports_root.exists():
            LOGGER.info("Reports root directory does not exist: %s", reports_root)
            return []

        # Assume each run has its own subdirectory under REPORTS_DIR
        run_ids: List[str] = [
            p.name for p in reports_root.iterdir()
            if p.is_dir()
        ]
        return sorted(run_ids)


def list_reports_for_run(
        run_id: str,
        prefix: str | None = None,
        suffix: str = ".pdf",
) -> List[str]:
    """
    List all report references for a given run_id.

    Returns:
        LOCAL → list of filesystem paths as str
        S3    → list of s3://bucket/key URIs
    """
    run_id = run_id.strip("/")

    if is_s3():
        bucket = getattr(SETTINGS, "REPORTS_BUCKET", None)
        if not bucket:
            LOGGER.warning("REPORTS_BUCKET not configured; list_reports_for_run → []")
            return []

        base_prefix = run_id
        search = f"{base_prefix}/"
        keys = [
            k for k in list_bucket_objects(bucket, search)
            if k.startswith(search)
               and k.endswith(suffix)
               and (prefix is None or Path(k).name.startswith(prefix))
        ]
        keys.sort()
        # Return full s3:// URIs so load_report_for_download can use them
        return [formulate_s3_uri(bucket, k) for k in keys]
    else:
        reports_dir = Path(SETTINGS.REPORTS_DIR) / run_id
        if not reports_dir.exists():
            LOGGER.info("Reports dir for run_id %s does not exist: %s", run_id, reports_dir)
            return []

        files = [
            p for p in reports_dir.iterdir()
            if p.is_file()
               and p.suffix == suffix
               and (prefix is None or p.name.startswith(prefix))
        ]
        files.sort(key=lambda p: p.name)
        return [str(p) for p in files]


# -----------------------------
# Prediction runs (single + batch)
# -----------------------------
def save_predictions(df: pd.DataFrame, base_dir: str, name: str) -> str:
    """
    Save prediction result dataset (single or batch) as Parquet.

    Uses PREDICTIONS_BUCKET / PREDICTIONS_DIR when configured,
    otherwise falls back to PROCESSED bucket/dir.
    """
    if is_s3():
        _ensure_buckets()
        bucket = getattr(SETTINGS, "PREDICTIONS_BUCKET", None) or SETTINGS.PROCESSED_BUCKET
        key = f"{base_dir.strip('/')}/{name}.parquet"
        write_dataframe_parquet(df, bucket, key, index=False)
        uri = formulate_s3_uri(bucket, key)
        LOGGER.info(f"Saved predictions (S3) → {uri}")
        return uri
    else:
        root = Path(getattr(SETTINGS, "PREDICTIONS_DIR", SETTINGS.PROCESSED_DIR))
        path = os.path.join(root / base_dir, f"{name}.parquet")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        LOGGER.info(f"Saved predictions (LOCAL) → {path}")
        return path


def save_prediction_metadata(meta: dict, base_dir: str, name: str) -> str:
    """
    Save prediction run metadata (JSON) alongside prediction datasets.

    Uses PREDICTIONS_BUCKET / PREDICTIONS_DIR when configured,
    otherwise falls back to PROCESSED bucket/dir.
    """
    filename = f"{name}.json"

    if is_s3():
        _ensure_buckets()
        bucket = getattr(SETTINGS, "PREDICTIONS_BUCKET", None) or SETTINGS.PROCESSED_BUCKET
        key = f"{base_dir.strip('/')}/{filename}"
        write_bucket_object(
            bucket,
            key,
            json.dumps(meta, indent=2),
            content_type="application/json",
        )
        uri = formulate_s3_uri(bucket, key)
        LOGGER.info(f"Saved prediction metadata (S3) → {uri}")
        return uri
    else:
        root = Path(getattr(SETTINGS, "PREDICTIONS_DIR", SETTINGS.PROCESSED_DIR))
        out_path = root / base_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        LOGGER.info(f"Saved prediction metadata (LOCAL) → {out_path}")
        return str(out_path)
