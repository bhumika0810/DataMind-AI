from __future__ import annotations

from typing import Any
import re

import pandas as pd

from services.analytics_engine.models import AnalysisPlan, EngineResult


def execute_plan(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> list[EngineResult]:
    results = []
    for engine_name in plan.engines:
        try:
            if engine_name == "statistics":
                results.append(statistics_engine(df, profile, plan))
            elif engine_name == "trend":
                results.append(trend_engine(df, profile, plan))
            elif engine_name == "correlation":
                results.append(correlation_engine(df, profile, plan))
            elif engine_name == "outlier":
                results.append(outlier_engine(df, profile, plan))
            elif engine_name == "data_quality":
                results.append(data_quality_engine(df, profile, plan))
            elif engine_name == "summarization":
                results.append(summarization_engine(df, profile, plan))
        except Exception as exc:
            results.append(EngineResult(engine_name, False, "", error=str(exc)))

    return results


def statistics_engine(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> EngineResult:
    question_l = plan.question.lower()

    count_result = _answer_count_filter_question(df, profile, plan)
    if count_result is not None:
        return count_result

    measure = _selected_measure(df, profile, plan)
    dimension = _selected_dimension(df, profile, plan, exclude={measure})

    generated_code = "df.copy()"
    metrics = {}
    insights = []
    data = None

    if measure:
        values = pd.to_numeric(df[measure], errors="coerce").dropna()
        metrics[measure] = {
            "count": int(values.count()),
            "mean": _round(values.mean()),
            "median": _round(values.median()),
            "mode": _safe_mode(values),
            "variance": _round(values.var()),
            "standard_deviation": _round(values.std()),
            "min": _round(values.min()),
            "max": _round(values.max()),
            "skewness": _round(values.skew()),
            "kurtosis": _round(values.kurt()),
            "p25": _round(values.quantile(0.25)),
            "p75": _round(values.quantile(0.75)),
        }

        if any(term in question_l for term in ["top", "highest", "largest", "best"]):
            data = df.nlargest(min(10, len(df)), measure).head(10).to_dict(orient="records")
            generated_code = f"df.nlargest(10, {measure!r})"
            insights.append(f"The highest {measure} value is {_round(values.max())}.")
        elif any(term in question_l for term in ["bottom", "lowest", "smallest", "worst"]):
            data = df.nsmallest(min(10, len(df)), measure).head(10).to_dict(orient="records")
            generated_code = f"df.nsmallest(10, {measure!r})"
            insights.append(f"The lowest {measure} value is {_round(values.min())}.")
        elif dimension:
            grouped = df.groupby(dimension, dropna=False)[measure].sum().sort_values(ascending=False).head(10)
            data = grouped.reset_index(name=measure).to_dict(orient="records")
            generated_code = f"df.groupby({dimension!r})[{measure!r}].sum().sort_values(ascending=False).head(10)"
            leader = grouped.index[0] if not grouped.empty else None
            if leader is not None:
                insights.append(f"{leader} leads on {measure} with {_round(grouped.iloc[0])}.")
        else:
            data = metrics[measure]
            generated_code = f"pd.to_numeric(df[{measure!r}], errors='coerce').describe()"
            insights.append(f"The average {measure} is {_round(values.mean())}, with median {_round(values.median())}.")
    else:
        data = {"row_count": len(df), "column_count": len(df.columns)}
        insights.append(f"The dataset contains {len(df)} rows across {len(df.columns)} columns.")

    return EngineResult("statistics", True, generated_code, data=data, metrics=metrics, insights=insights)


def _answer_count_filter_question(
    df: pd.DataFrame,
    profile: dict[str, Any],
    plan: AnalysisPlan,
) -> EngineResult | None:
    question_l = plan.question.lower()
    asks_count = any(term in question_l for term in ["how many", "number of", "count"])
    if not asks_count:
        return None

    filter_match = _find_categorical_filter(df, profile, question_l)
    if not filter_match:
        return None

    filter_column, filter_value = filter_match
    mask = df[filter_column].astype(str).str.lower().str.strip() == str(filter_value).lower().strip()
    filtered = df[mask]

    entity_column = _selected_entity_id(df, profile, plan)
    if entity_column:
        count = int(filtered[entity_column].nunique(dropna=True))
        total = int(df[entity_column].nunique(dropna=True))
        entity_label = _humanize_column(entity_column)
        count_phrase = f"{count} {entity_label}"
        generated_code = (
            f"df[df[{filter_column!r}].astype(str).str.lower().str.strip() == "
            f"{str(filter_value).lower().strip()!r}][{entity_column!r}].nunique()"
        )
    else:
        count = int(len(filtered))
        total = int(len(df))
        entity_label = "records"
        count_phrase = f"{count} records"
        generated_code = (
            f"len(df[df[{filter_column!r}].astype(str).str.lower().str.strip() == "
            f"{str(filter_value).lower().strip()!r}])"
        )

    percentage = round((count / total) * 100, 2) if total else 0.0
    display_value = str(filter_value)
    insights = [
        f"There are {count_phrase} with {filter_column} = {display_value}.",
        f"That represents {percentage}% of the {total} total {entity_label}.",
    ]

    return EngineResult(
        "statistics",
        True,
        generated_code,
        data={
            "filter_column": filter_column,
            "filter_value": display_value,
            "count": count,
            "total": total,
            "percentage": percentage,
            "sample_rows": filtered.head(5).to_dict(orient="records"),
        },
        metrics={
            "matched_count": count,
            "total_count": total,
            "percentage": percentage,
            "filter": {
                "column": filter_column,
                "value": display_value,
            },
        },
        insights=insights,
    )


def trend_engine(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> EngineResult:
    date_col = _selected_date(profile, plan)
    measure = _selected_measure(df, profile, plan)
    if not date_col or not measure:
        return EngineResult("trend", False, "", error="Trend analysis requires one date column and one numeric measure.")

    working = df[[date_col, measure]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working[measure] = pd.to_numeric(working[measure], errors="coerce")
    working = working.dropna()

    if working.empty:
        return EngineResult("trend", False, "", error="No valid time series rows were available.")

    series = working.set_index(date_col).sort_index()[measure].resample("ME").sum()
    growth = series.pct_change().replace([float("inf"), -float("inf")], pd.NA)
    data = pd.DataFrame({
        "period": series.index.astype(str),
        measure: series.values,
        "growth_rate": growth.values,
        "moving_average_3": series.rolling(3, min_periods=1).mean().values,
    }).tail(12).to_dict(orient="records")

    total_change = _round(((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100) if len(series) > 1 and series.iloc[0] else None
    insights = [f"{measure} moved from {_round(series.iloc[0])} to {_round(series.iloc[-1])} across the available period."]
    if total_change is not None:
        insights.append(f"Overall period change was {total_change}%.")

    return EngineResult(
        "trend",
        True,
        f"df.set_index({date_col!r})[{measure!r}].resample('ME').sum()",
        data=data,
        metrics={"periods": int(len(series)), "overall_change_pct": total_change},
        insights=insights,
    )


def correlation_engine(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> EngineResult:
    numeric_cols = profile.get("semantic_roles", {}).get("possible_measures", [])
    if len(numeric_cols) < 2:
        return EngineResult("correlation", False, "", error="Correlation analysis requires at least two numeric columns.")

    corr = df[numeric_cols].corr(numeric_only=True).round(4)
    pairs = []
    for left in corr.columns:
        for right in corr.columns:
            if left >= right:
                continue
            pairs.append({"column_a": left, "column_b": right, "correlation": corr.loc[left, right]})
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    top = pairs[:10]

    insights = []
    if top:
        strongest = top[0]
        insights.append(
            f"The strongest numeric relationship is {strongest['column_a']} vs {strongest['column_b']} "
            f"with correlation {_round(strongest['correlation'])}."
        )

    return EngineResult(
        "correlation",
        True,
        f"df[{numeric_cols!r}].corr(numeric_only=True)",
        data=top,
        metrics={"correlation_matrix": corr.to_dict()},
        insights=insights,
    )


def outlier_engine(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> EngineResult:
    measure = _selected_measure(df, profile, plan)
    if not measure:
        return EngineResult("outlier", False, "", error="Outlier detection requires a numeric measure.")

    values = pd.to_numeric(df[measure], errors="coerce")
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(values < lower) | (values > upper)].head(20)

    return EngineResult(
        "outlier",
        True,
        f"df[(df[{measure!r}] < {lower}) | (df[{measure!r}] > {upper})]",
        data=outliers.to_dict(orient="records"),
        metrics={"measure": measure, "lower_bound": _round(lower), "upper_bound": _round(upper), "outlier_count": int(len(outliers))},
        insights=[f"Detected {len(outliers)} potential outlier records for {measure} using IQR thresholds."],
    )


def data_quality_engine(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> EngineResult:
    missing = profile.get("missing_values", {})
    high_missing = {
        column: meta
        for column, meta in missing.items()
        if meta.get("percentage", 0) >= 10
    }
    duplicate_count = profile.get("duplicates", {}).get("count", 0)
    insights = [
        f"Duplicate rows: {duplicate_count}.",
        f"Columns with 10%+ missing values: {len(high_missing)}.",
    ]

    return EngineResult(
        "data_quality",
        True,
        "df.isna().sum(); df.duplicated().sum()",
        data={"high_missing_columns": high_missing, "duplicates": profile.get("duplicates", {})},
        metrics={"duplicate_count": duplicate_count, "high_missing_column_count": len(high_missing)},
        insights=insights,
    )


def summarization_engine(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> EngineResult:
    roles = profile.get("semantic_roles", {})
    insights = [
        f"The dataset has {len(df)} rows and {len(df.columns)} columns.",
        f"Detected {len(roles.get('possible_measures', []))} measures, {len(roles.get('possible_dimensions', []))} dimensions and {len(roles.get('possible_dates', []))} date columns.",
    ]
    return EngineResult(
        "summarization",
        True,
        "generate_dataset_profile(df)",
        data={"roles": roles, "preview": profile.get("preview", [])},
        metrics={"rows": len(df), "columns": len(df.columns)},
        insights=insights,
    )


def _selected_measure(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> str | None:
    for concept in ["revenue", "salary", "cost", "profit", "quantity", "measure"]:
        match = plan.column_matches.get(concept)
        if match and match.column and pd.api.types.is_numeric_dtype(df[match.column]):
            return match.column

    measures = profile.get("semantic_roles", {}).get("possible_measures", [])
    return measures[0] if measures else None


def _find_categorical_filter(
    df: pd.DataFrame,
    profile: dict[str, Any],
    question_l: str,
) -> tuple[str, Any] | None:
    categorical_columns = profile.get("semantic_roles", {}).get("possible_dimensions", [])
    preferred = sorted(
        categorical_columns,
        key=lambda column: 0 if any(term in str(column).lower() for term in ["status", "state", "stage"]) else 1,
    )

    for column in preferred:
        values = df[column].dropna().astype(str).unique().tolist()
        for value in values:
            value_l = str(value).lower().strip()
            if not value_l or len(value_l) > 60:
                continue
            if re.search(rf"\b{re.escape(value_l)}\b", question_l):
                return column, value

    return None


def _selected_entity_id(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan) -> str | None:
    order_match = plan.column_matches.get("order")
    if order_match and order_match.column:
        return order_match.column

    question_l = plan.question.lower()
    ids = profile.get("semantic_roles", {}).get("possible_ids", [])
    if "order" in question_l:
        for column in ids:
            column_l = str(column).lower().replace("_", " ")
            if "order" in column_l:
                return column

    primary_keys = profile.get("semantic_roles", {}).get("possible_primary_keys", [])
    return primary_keys[0] if primary_keys else None


def _humanize_column(column: str) -> str:
    label = str(column).replace("_", " ").strip()
    label_l = label.lower()
    if label_l.endswith(" id"):
        label_l = label_l[:-3]
    if not label_l.endswith("s"):
        label_l += "s"
    return label_l


def _selected_dimension(df: pd.DataFrame, profile: dict[str, Any], plan: AnalysisPlan, exclude: set[str | None]) -> str | None:
    for concept in ["department", "region", "product", "customer", "employee", "status"]:
        match = plan.column_matches.get(concept)
        if match and match.column and match.column not in exclude:
            return match.column

    dimensions = [col for col in profile.get("semantic_roles", {}).get("possible_dimensions", []) if col not in exclude]
    return dimensions[0] if dimensions else None


def _selected_date(profile: dict[str, Any], plan: AnalysisPlan) -> str | None:
    match = plan.column_matches.get("date")
    if match and match.column:
        return match.column
    dates = profile.get("semantic_roles", {}).get("possible_dates", [])
    return dates[0] if dates else None


def _round(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
        return round(float(value), 4)
    except Exception:
        return value


def _safe_mode(values: pd.Series) -> Any:
    try:
        mode = values.mode()
        return _round(mode.iloc[0]) if not mode.empty else None
    except Exception:
        return None
