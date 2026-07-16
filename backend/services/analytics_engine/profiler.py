from __future__ import annotations

from typing import Any
import warnings

import pandas as pd

from services.dataset_profiler import generate_dataset_profile


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in normalized.columns:
        series = normalized[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(1, int(series.notna().sum() * 0.7)):
            normalized[column] = numeric
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed_dates = pd.to_datetime(series, errors="coerce")
        if parsed_dates.notna().sum() >= max(1, int(series.notna().sum() * 0.8)):
            normalized[column] = parsed_dates

    return normalized


def build_enterprise_profile(df: pd.DataFrame) -> dict[str, Any]:
    profile = generate_dataset_profile(df)
    row_count = len(df)

    possible_ids = []
    possible_measures = []
    possible_dimensions = []
    possible_dates = []

    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        unique_ratio = unique / non_null if non_null else 0
        name_l = str(column).lower()

        if unique_ratio > 0.9 or name_l.endswith("id") or "_id" in name_l or "code" in name_l:
            possible_ids.append(str(column))

        if pd.api.types.is_numeric_dtype(series):
            possible_measures.append(str(column))
        elif pd.api.types.is_datetime64_any_dtype(series):
            possible_dates.append(str(column))
        else:
            possible_dimensions.append(str(column))

    profile["semantic_roles"] = {
        "possible_primary_keys": [
            col for col in possible_ids
            if df[col].nunique(dropna=True) == row_count and row_count > 0
        ],
        "possible_ids": possible_ids,
        "possible_measures": possible_measures,
        "possible_dimensions": possible_dimensions,
        "possible_dates": possible_dates,
        "possible_foreign_keys": [
            col for col in possible_ids
            if col not in possible_measures and df[col].nunique(dropna=True) < row_count
        ],
    }

    return profile
