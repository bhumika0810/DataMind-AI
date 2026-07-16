from __future__ import annotations

from typing import Any
import warnings

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def _human_readable_bytes(size_in_bytes: int) -> str:
    """Convert a byte count into a compact human-readable value."""
    try:
        size = float(size_in_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} {unit}"
            size /= 1024
    except Exception:
        return "0 B"

    return "0 B"


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation for pandas/numpy values."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass

    return value


def _series_as_json_values(series: pd.Series, limit: int = 5) -> list[Any]:
    """Return non-null sample values from a Series as JSON-safe Python values."""
    try:
        return [_json_safe(value) for value in series.dropna().head(limit).tolist()]
    except Exception:
        return []


def _looks_like_datetime(series: pd.Series) -> bool:
    """Detect date-like object/string columns without forcing every text column into dates."""
    try:
        if is_datetime64_any_dtype(series):
            return True
        if not (is_object_dtype(series) or is_string_dtype(series)):
            return False

        sample = series.dropna().astype(str).head(25)
        if sample.empty:
            return False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.8
    except Exception:
        return False


def _column_category(series: pd.Series) -> str:
    """Infer the high-level category used by the dataset profile."""
    try:
        if is_bool_dtype(series):
            return "boolean"
        if _looks_like_datetime(series):
            return "datetime"
        if is_numeric_dtype(series):
            return "numeric"
        if is_object_dtype(series) or is_string_dtype(series):
            unique_count = series.nunique(dropna=True)
            non_null_count = int(series.notna().sum())
            if non_null_count and unique_count / non_null_count > 0.5:
                return "text"
            return "categorical"
    except Exception:
        pass

    return "text"


def _numeric_statistics(series: pd.Series) -> dict[str, Any]:
    """Safely calculate numeric summary statistics for one column."""
    try:
        numeric_series = pd.to_numeric(series, errors="coerce")
        return {
            "min": _json_safe(numeric_series.min()),
            "max": _json_safe(numeric_series.max()),
            "mean": _json_safe(numeric_series.mean()),
            "median": _json_safe(numeric_series.median()),
            "std": _json_safe(numeric_series.std()),
        }
    except Exception:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }


def _categorical_statistics(series: pd.Series) -> dict[str, Any]:
    """Safely summarize categorical or free-text values."""
    try:
        top_values = series.dropna().value_counts().head(5)
        return {
            "unique_values": int(series.nunique(dropna=True)),
            "top_values": {
                str(_json_safe(index)): int(count)
                for index, count in top_values.items()
            },
            "sample_values": _series_as_json_values(series),
        }
    except Exception:
        return {
            "unique_values": 0,
            "top_values": {},
            "sample_values": [],
        }


def _datetime_statistics(series: pd.Series) -> dict[str, Any]:
    """Safely calculate date range statistics for one column."""
    try:
        datetime_series = pd.to_datetime(series, errors="coerce")
        return {
            "min": _json_safe(datetime_series.min()),
            "max": _json_safe(datetime_series.max()),
        }
    except Exception:
        return {
            "min": None,
            "max": None,
        }


def _preview_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert the first five rows to JSON-safe dictionaries."""
    try:
        records = df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records")
        return [
            {str(key): _json_safe(value) for key, value in record.items()}
            for record in records
        ]
    except Exception:
        return []


def generate_dataset_profile(df: pd.DataFrame) -> dict:
    """
    Generate a rich, JSON-serializable profile for a pandas DataFrame.

    The profiler is defensive by design: a problematic column should produce
    empty/default statistics for that column instead of failing the full profile.
    """
    if not isinstance(df, pd.DataFrame):
        return {
            "general": {"rows": 0, "columns": 0, "memory_usage": "0 B"},
            "columns": [],
            "numeric_statistics": {},
            "categorical_statistics": {},
            "datetime_statistics": {},
            "missing_values": {},
            "duplicates": {"count": 0, "percentage": 0.0},
            "summary": {
                "numeric_columns": [],
                "categorical_columns": [],
                "datetime_columns": [],
                "boolean_columns": [],
            },
            "preview": [],
        }

    row_count = int(len(df))
    column_count = int(len(df.columns))

    try:
        memory_usage = _human_readable_bytes(int(df.memory_usage(deep=True).sum()))
    except Exception:
        memory_usage = "0 B"

    profile = {
        "general": {
            "rows": row_count,
            "columns": column_count,
            "memory_usage": memory_usage,
        },
        "columns": [],
        "numeric_statistics": {},
        "categorical_statistics": {},
        "datetime_statistics": {},
        "missing_values": {},
        "duplicates": {"count": 0, "percentage": 0.0},
        "summary": {
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "boolean_columns": [],
        },
        "preview": _preview_records(df),
    }

    for column in df.columns:
        column_name = str(column)

        try:
            series = df[column]
            dtype = str(series.dtype)
            category = _column_category(series)
        except Exception:
            series = pd.Series(dtype="object")
            dtype = "unknown"
            category = "text"

        profile["columns"].append(
            {
                "name": column_name,
                "dtype": dtype,
                "category": category,
            }
        )

        try:
            missing_count = int(series.isna().sum())
            missing_percentage = round((missing_count / row_count) * 100, 2) if row_count else 0.0
        except Exception:
            missing_count = 0
            missing_percentage = 0.0

        profile["missing_values"][column_name] = {
            "count": missing_count,
            "percentage": missing_percentage,
        }

        if category == "numeric":
            profile["summary"]["numeric_columns"].append(column_name)
            profile["numeric_statistics"][column_name] = _numeric_statistics(series)
        elif category == "datetime":
            profile["summary"]["datetime_columns"].append(column_name)
            profile["datetime_statistics"][column_name] = _datetime_statistics(series)
        elif category == "boolean":
            profile["summary"]["boolean_columns"].append(column_name)
        elif category in {"categorical", "text"}:
            profile["summary"]["categorical_columns"].append(column_name)
            profile["categorical_statistics"][column_name] = _categorical_statistics(series)

    try:
        duplicate_count = int(df.duplicated().sum())
        duplicate_percentage = round((duplicate_count / row_count) * 100, 2) if row_count else 0.0
    except Exception:
        duplicate_count = 0
        duplicate_percentage = 0.0

    profile["duplicates"] = {
        "count": duplicate_count,
        "percentage": duplicate_percentage,
    }

    return profile
