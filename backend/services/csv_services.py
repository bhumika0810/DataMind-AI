"""
services/csv_service.py
Handles: uploading (CSV + Excel), reading, parsing, and metadata extraction.
"""

import os
import uuid
import hashlib
import pandas as pd
from pathlib import Path
from typing import Any
from config import settings


# Supported upload extensions → how to read them
_READERS = {
    ".csv":  lambda p: pd.read_csv(p),
    ".xlsx": lambda p: pd.read_excel(p, engine="openpyxl"),
    ".xls":  lambda p: pd.read_excel(p, engine="xlrd"),
}


# ─────────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────────

async def save_csv(file_bytes: bytes, original_filename: str) -> dict[str, Any]:
    """
    Save uploaded CSV or Excel bytes to disk (always stored as .csv).
    Returns file_id, path, and basic info.
    """
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    ext = Path(original_filename).suffix.lower()
    if ext not in _READERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {list(_READERS)}")

    # Unique filename — always store as CSV internally
    file_id   = str(uuid.uuid4())
    safe_name = Path(original_filename).stem[:40]
    raw_path  = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{safe_name}{ext}")

    with open(raw_path, "wb") as f:
        f.write(file_bytes)

    # Parse and re-save as .csv (normalises Excel files too)
    try:
        df = _READERS[ext](raw_path)
    except Exception as e:
        os.remove(raw_path)
        raise ValueError(f"Could not parse file: {e}")

    # Clean column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    csv_filename = f"{file_id}_{safe_name}.csv"
    csv_path     = os.path.join(settings.UPLOAD_DIR, csv_filename)
    df.to_csv(csv_path, index=False)

    # Remove the raw upload if it was Excel
    if ext != ".csv":
        os.remove(raw_path)

    return {
        "file_id":           file_id,
        "filename":          csv_filename,
        "original_filename": original_filename,
        "filepath":          csv_path,
        "size_bytes":        len(file_bytes),
        "size_kb":           round(len(file_bytes) / 1024, 2),
        "columns":           list(df.columns),
        "row_count":         len(df),
    }


# ─────────────────────────────────────────────
#  READ
# ─────────────────────────────────────────────

def read_csv(file_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    """
    Read rows from an uploaded CSV with pagination.
    """
    filepath = _find_file(file_id)
    df       = pd.read_csv(filepath)
    total    = len(df)
    page     = df.iloc[offset : offset + limit]

    return {
        "file_id":     file_id,
        "total_rows":  total,
        "offset":      offset,
        "limit":       limit,
        "columns":     list(df.columns),
        "data":        page.to_dict(orient="records"),
        "returned":    len(page),
    }


# ─────────────────────────────────────────────
#  METADATA EXTRACTION
# ─────────────────────────────────────────────

def extract_metadata(file_id: str) -> dict[str, Any]:
    """
    Deep metadata extraction:
    - shape, dtypes, nulls, uniques, numeric stats, sample values
    """
    filepath = _find_file(file_id)
    df       = pd.read_csv(filepath)

    # File hash for integrity
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    columns_meta = []
    for col in df.columns:
        series      = df[col]
        dtype       = str(series.dtype)
        null_count  = int(series.isna().sum())
        unique      = int(series.nunique(dropna=True))
        sample_vals = series.dropna().head(5).tolist()

        col_info: dict[str, Any] = {
            "name":         col,
            "dtype":        dtype,
            "null_count":   null_count,
            "null_pct":     round(null_count / len(df) * 100, 2) if len(df) else 0,
            "unique_count": unique,
            "sample_values": sample_vals,
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info["stats"] = {
                "min":    round(float(desc["min"]),  4),
                "max":    round(float(desc["max"]),  4),
                "mean":   round(float(desc["mean"]), 4),
                "median": round(float(series.median()), 4),
                "std":    round(float(desc["std"]),  4),
            }

        # Date detection
        if dtype == "object":
            try:
                pd.to_datetime(series.dropna().head(20), infer_datetime_format=True)
                col_info["likely_date"] = True
            except Exception:
                col_info["likely_date"] = False

        columns_meta.append(col_info)

    # Duplicate rows
    dup_count = int(df.duplicated().sum())

    return {
        "file_id":          file_id,
        "file_hash_md5":    file_hash,
        "shape": {
            "rows":    len(df),
            "columns": len(df.columns),
        },
        "column_count":     len(df.columns),
        "row_count":        len(df),
        "duplicate_rows":   dup_count,
        "memory_usage_kb":  round(df.memory_usage(deep=True).sum() / 1024, 2),
        "columns":          columns_meta,
    }


# ─────────────────────────────────────────────
#  LIST UPLOADED FILES
# ─────────────────────────────────────────────

def list_files(limit: int = 20) -> list[dict]:
    """Return uploaded CSV files with basic info, newest first."""
    upload_dir = settings.UPLOAD_DIR
    if not os.path.exists(upload_dir):
        return []

    files = []
    for fname in os.listdir(upload_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(upload_dir, fname)
        size  = os.path.getsize(fpath)
        file_id = fname.split("_")[0]

        with open(fpath, "rb") as f:
            row_count = max(sum(1 for _ in f) - 1, 0)

        mtime = os.path.getmtime(fpath)
        files.append({
            "file_id":           file_id,
            "filename":          fname,
            "size_kb":           round(size / 1024, 2),
            "original_filename": "_".join(fname.split("_")[1:]),
            "row_count":         row_count,
            "uploaded_at":       mtime,
        })

    files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return files[:limit]


# ─────────────────────────────────────────────
#  DELETE FILE
# ─────────────────────────────────────────────

def delete_file(file_id: str) -> dict:
    """Delete an uploaded CSV by file_id."""
    filepath = _find_file(file_id)
    os.remove(filepath)
    return {"deleted": True, "file_id": file_id}


# ─────────────────────────────────────────────
#  INTERNAL HELPER
# ─────────────────────────────────────────────

def _find_file(file_id: str) -> str:
    """Locate a file on disk by its UUID prefix."""
    upload_dir = settings.UPLOAD_DIR
    if not os.path.exists(upload_dir):
        raise FileNotFoundError(f"Upload directory '{upload_dir}' does not exist.")

    for fname in os.listdir(upload_dir):
        if fname.startswith(file_id) and fname.endswith(".csv"):
            return os.path.join(upload_dir, fname)

    raise FileNotFoundError(f"No file found with file_id: {file_id}")