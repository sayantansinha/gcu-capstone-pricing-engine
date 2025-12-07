import io
import json
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional

import boto3
import pandas as pd
import s3fs
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from src.config.env_loader import SETTINGS
from src.config.s3_constants import S3_EXPECTED_BUCKET_OWNER, S3_ADDRESSING_STYLE
from src.utils.log_utils import get_logger

# ---------------------------------------------------------------------
# Logger setup
LOGGER = get_logger("s3_utils")

# ---------------------------------------------------------------------
# boto3 client (honors AWS_ENDPOINT_URL for LocalStack/MinIO; leave unset for AWS)
_session = boto3.session.Session()
s3 = _session.client(
    "s3",
    region_name=SETTINGS.AWS_REGION,
    endpoint_url=SETTINGS.AWS_ENDPOINT_URL or None,
    config=Config(signature_version="s3v4"),
)


# ---------------------------------------------------------------------
def formulate_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def s3opts() -> Dict[str, Any]:
    client_kwargs: Dict[str, Any] = {}
    if SETTINGS.AWS_ENDPOINT_URL:
        client_kwargs["endpoint_url"] = SETTINGS.AWS_ENDPOINT_URL
    if SETTINGS.AWS_REGION:
        client_kwargs["region_name"] = SETTINGS.AWS_REGION

    config_kwargs: Dict[str, Any] = {}
    if S3_ADDRESSING_STYLE:
        config_kwargs["s3"] = {"addressing_style": S3_ADDRESSING_STYLE}

    key_kwargs: Dict[str, Any] = {}
    if SETTINGS.AWS_ACCESS_KEY_ID and SETTINGS.AWS_SECRET_ACCESS_KEY:
        key_kwargs.update({"key": SETTINGS.AWS_ACCESS_KEY_ID, "secret": SETTINGS.AWS_SECRET_ACCESS_KEY})
        if SETTINGS.AWS_SESSION_TOKEN:
            key_kwargs["token"] = SETTINGS.AWS_SESSION_TOKEN

    storage_options: Dict[str, Any] = {}
    if client_kwargs:
        storage_options["client_kwargs"] = client_kwargs
    if config_kwargs:
        storage_options["config_kwargs"] = config_kwargs
    if key_kwargs:
        storage_options.update(key_kwargs)
    if S3_EXPECTED_BUCKET_OWNER:
        storage_options["expected_bucket_owner"] = S3_EXPECTED_BUCKET_OWNER

    return storage_options


def fetch_s3fs() -> s3fs.S3FileSystem:
    """Unified s3fs constructor for AWS or LocalStack."""
    return s3fs.S3FileSystem(**s3opts())


# ---------------------------------------------------------------------
def _owner_src() -> dict:
    return {"ExpectedSourceBucketOwner": S3_EXPECTED_BUCKET_OWNER} if S3_EXPECTED_BUCKET_OWNER else {}


def _owner_dst() -> dict:
    return {"ExpectedBucketOwner": S3_EXPECTED_BUCKET_OWNER} if S3_EXPECTED_BUCKET_OWNER else {}


def _content_type(path: Path) -> Optional[str]:
    ctype, _ = mimetypes.guess_type(str(path))
    if path.suffix.lower() == ".parquet":
        return "application/octet-stream"
    return ctype


def _put_object_extra_args(
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        sse: Optional[str] = None,
        sse_kms_key_id: Optional[str] = None,
        storage_class: Optional[str] = None,
        acl: Optional[str] = None,
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if content_type:
        extra["ContentType"] = content_type
    if metadata:
        extra["Metadata"] = metadata
    if sse:
        extra["ServerSideEncryption"] = sse
    if sse_kms_key_id:
        extra["SSEKMSKeyId"] = sse_kms_key_id
    if storage_class:
        extra["StorageClass"] = storage_class
    if acl:
        extra["ACL"] = acl
    return extra


# ---------------------------------------------------------------------
# Buckets & objects
def ensure_bucket(name: str):
    """
    Ensure bucket exists. Handles non-us-east-1 create (needs LocationConstraint).
    """
    try:
        kwargs = {"Bucket": name}
        if S3_EXPECTED_BUCKET_OWNER:
            kwargs["ExpectedBucketOwner"] = S3_EXPECTED_BUCKET_OWNER
        s3.head_bucket(**kwargs)
        return
    except ClientError:
        pass

    LOGGER.info(f"Creating bucket: s3://{name}")
    create_kwargs = {"Bucket": name}
    # LocalStack often ignores LocationConstraint; real AWS requires it if region != us-east-1
    if not SETTINGS.AWS_ENDPOINT_URL and SETTINGS.AWS_REGION and SETTINGS.AWS_REGION != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": SETTINGS.AWS_REGION}
    s3.create_bucket(**create_kwargs)


def list_bucket_objects(bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    base_params = {"Bucket": bucket, "Prefix": prefix, **_owner_dst()}
    for page in paginator.paginate(**base_params):
        for o in page.get("Contents", []):
            k = o["Key"]
            if not k.endswith("/"):
                keys.append(k)
    return keys


def upload_file_to_bucket(bucket: str, key: str, local: Path):
    extra_upload: Dict[str, Any] = {}
    ctype = _content_type(local)
    if ctype:
        extra_upload["ContentType"] = ctype
    extra_upload.update(_owner_dst())
    LOGGER.info(f"{local}  →  {formulate_s3_uri(bucket, key)}")
    s3.upload_file(str(local), bucket, key, ExtraArgs=extra_upload)


def copy_bucket_object(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str):
    args = {
        "Bucket": dst_bucket,
        "CopySource": {"Bucket": src_bucket, "Key": src_key},
        "Key": dst_key,
        **_owner_dst(),
        **_owner_src()
    }
    s3.copy_object(**args)


def delete_bucket_object(bucket: str, key: str):
    LOGGER.info(f"DELETE  →  {formulate_s3_uri(bucket, key)}")
    s3.delete_object(Bucket=bucket, Key=key, **_owner_dst())


def load_bucket_object(bucket: str, key: str) -> Any:
    """
    Load an S3 object into a Python object.

    - *.parquet → pandas.DataFrame
    - *.csv / *.tsv → pandas.DataFrame
    - *.json → dict / list (parsed via json.loads)
    - all other types → raw bytes (for binary files like joblib)
    """
    k = key.lower()

    # Parquet via s3fs
    if k.endswith(".parquet"):
        fs = fetch_s3fs()
        return pd.read_parquet(formulate_s3_uri(bucket, key), filesystem=fs)

    # Generic object via GetObject
    params = {"Bucket": bucket, "Key": key, **_owner_dst()}
    try:
        obj = s3.get_object(**params)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        # Normalize S3 "not found" into FileNotFoundError
        if error_code in ("NoSuchKey", "NotFound", "404"):
            LOGGER.info(f"Object not found: {formulate_s3_uri(bucket, key)}")
            raise FileNotFoundError(f"s3://{bucket}/{key} not found") from e
        LOGGER.exception(f"ClientError fetching s3://{bucket}/{key}: {e}")
        raise

    body = obj["Body"].read()

    # JSON path
    if k.endswith(".json"):
        try:
            return json.loads(body.decode("utf-8"))
        except Exception as ex:
            LOGGER.exception(f"Error parsing JSON for s3://{bucket}/{key}: {ex}")
            raise

    # CSV / TSV path → DataFrame
    if k.endswith(".csv") or k.endswith(".tsv"):
        sep = "\t" if k.endswith(".tsv") else ","
        return pd.read_csv(io.BytesIO(body), sep=sep)

    # All other types → return raw bytes (e.g., for joblib / pickle)
    return body


def write_bucket_object(
        bucket: str,
        key: str,
        payload: bytes | str,
        *,
        content_type: Optional[str] = None,
        encoding: str = "utf-8",
        metadata: Optional[Dict[str, str]] = None,
        sse: Optional[str] = None,
        sse_kms_key_id: Optional[str] = None,
        storage_class: Optional[str] = None,
        acl: Optional[str] = None,
) -> str:
    body: bytes = payload if isinstance(payload, bytes) else payload.encode(encoding)
    params: Dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        **_put_object_extra_args(
            content_type=content_type,
            metadata=metadata,
            sse=sse,
            sse_kms_key_id=sse_kms_key_id,
            storage_class=storage_class,
            acl=acl,
        ),
        **_owner_dst(),
    }
    LOGGER.info(f"PUT  →  {formulate_s3_uri(bucket, key)}  ({len(body)} bytes)")
    resp = s3.put_object(**params)
    return (resp or {}).get("ETag") or formulate_s3_uri(bucket, key)


# ---------------------------------------------------------------------
# Convenience helpers you’ll use from Streamlit pages
def write_dataframe_parquet(df: pd.DataFrame, bucket: str, key: str, *, index: bool = False):
    """Write a DataFrame as Parquet via s3fs (recommended for speed/size)."""
    fs = fetch_s3fs()
    uri = formulate_s3_uri(bucket, key)
    LOGGER.info(f"PARQUET PUT → {uri}  (rows={len(df)})")
    with fs.open(uri, "wb") as f:
        df.to_parquet(f, index=index)


def write_dataframe_csv(df: pd.DataFrame, bucket: str, key: str, *, index: bool = False):
    """Write a DataFrame as CSV (for quick inspection / downloads)."""
    fs = fetch_s3fs()
    uri = formulate_s3_uri(bucket, key)
    LOGGER.info(f"CSV PUT → {uri}  (rows={len(df)})")
    with fs.open(uri, "w") as f:
        df.to_csv(f, index=index)


def upload_directory(local_dir: Path, bucket: str, prefix: str = ""):
    """Recursively upload a directory (small class datasets / artifacts)."""
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            upload_file_to_bucket(bucket, key, p)


def download_file(bucket: str, key: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"GET  ←  {formulate_s3_uri(bucket, key)}  → {local_path}")
    s3.download_file(bucket, key, str(local_path), ExtraArgs=_owner_dst() or {})


def generate_presigned_url(bucket: str, key: str, expires_in_seconds: int = 900) -> str:
    """Use for report previews / controlled downloads."""
    try:
        return s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key, **_owner_dst()},
            ExpiresIn=expires_in_seconds,
        )
    except (ClientError, NoCredentialsError) as e:
        LOGGER.error(f"Failed to sign URL for s3://{bucket}/{key}: {e}")
        return ""
