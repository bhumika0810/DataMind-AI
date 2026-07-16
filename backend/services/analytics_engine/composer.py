from __future__ import annotations

from dataclasses import asdict
from typing import Any
import re

import pandas as pd

from services.analytics_engine.charting import generate_chart
from services.analytics_engine.models import AnalysisPlan, AnalyticsResponse, EngineResult


def compose_response(
    file_id: str,
    question: str,
    df: pd.DataFrame,
    profile: dict[str, Any],
    plan: AnalysisPlan,
    results: list[EngineResult],
    execution_time_ms: int,
) -> AnalyticsResponse:
    response_mode = plan.intents[0] if plan.intents else "quick_answer"
    successful = [result for result in results if result.success]
    insights_text = [insight for result in successful for insight in result.insights]
    statistics = _statistics_for_mode(response_mode, successful)
    charts = _chart_suggestions(plan, successful) if response_mode in {"visualization", "executive_report"} else []
    chart_error = None
    if response_mode == "visualization":
        chart_result = generate_chart(question, df, plan)
        if chart_result.get("success"):
            charts = [{
                "chart_type": chart_result["chart_type"],
                "title": chart_result["title"],
                "image_url": chart_result["image_url"],
                "image_path": chart_result["image_path"],
            }]
        else:
            charts = []
            chart_error = chart_result.get("error") or "Could not generate the requested chart."
    recommendations = _recommendations(plan, results, profile) if response_mode in {"recommendation", "executive_report"} else []
    confidence = _confidence(plan, results) if response_mode == "executive_report" else None

    if plan.clarification:
        business_summary = plan.clarification
    elif insights_text:
        business_summary = " ".join(insights_text[:3])
    else:
        business_summary = "The dataset was profiled, but no strong analytical result could be computed for this question."

    if response_mode == "quick_answer":
        answer = _quick_answer(question, df, profile, plan, successful, business_summary)
        business_summary = ""
        insights = []
    else:
        answer = _format_answer(
            response_mode=response_mode,
            plan=plan,
            business_summary=business_summary,
            statistics=statistics,
            charts=charts,
            insights_text=insights_text,
            recommendations=recommendations,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            chart_error=chart_error,
        )

        insights = [{"type": "text", "content": text} for text in insights_text]
        for result in successful:
            if isinstance(result.data, list) and result.data:
                insights.append({
                    "type": "table",
                    "content": {
                        "title": f"{result.name.title()} Result",
                        "rows": _json_safe_records(result.data[:10]),
                    },
                })
        for chart in charts:
            insights.append({"type": "chart", "content": chart})

        if response_mode != "executive_report":
            business_summary = ""

    return AnalyticsResponse(
        answer=answer,
        business_summary=business_summary,
        statistics=statistics,
        charts=charts,
        insights=insights,
        recommendations=recommendations,
        confidence_score=confidence,
        execution_time_ms=execution_time_ms,
        analysis_plan=_plan_to_dict(plan),
        generated_code=[result.generated_code for result in results if result.generated_code],
        suggested_followups=_followups(plan, profile) if response_mode == "executive_report" else [],
    )


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)


def _statistics_for_mode(response_mode: str, results: list[EngineResult]) -> dict[str, Any]:
    if response_mode == "quick_answer":
        return {}
    return {result.name: result.metrics for result in results if result.metrics}


def _format_answer(
    response_mode: str,
    plan: AnalysisPlan,
    business_summary: str,
    statistics: dict[str, Any],
    charts: list[dict[str, Any]],
    insights_text: list[str],
    recommendations: list[str],
    confidence: float | None,
    execution_time_ms: int,
    chart_error: str | None = None,
) -> str:
    if response_mode == "executive_report":
        lines = [
            "Answer",
            business_summary,
            "",
            "Business Summary",
            f"Domain interpreted as {plan.domain}. Intent detected: {', '.join(plan.intents)}.",
            "",
            "Statistics",
            _compact_statistics(statistics),
            "",
            "Charts",
            _compact_charts(charts),
            "",
            "Insights",
            _bullet(insights_text or ["No deeper insight was available from the selected analytical engines."]),
            "",
            "Recommendations",
            _bullet(recommendations),
            "",
            "Confidence Score",
            f"{confidence:.2f}" if confidence is not None else "Not calculated",
            "",
            "Execution Time",
            f"{execution_time_ms} ms",
        ]
        return "\n".join(lines)

    if response_mode == "recommendation":
        lines = [
            "Answer",
            business_summary,
            "",
            "Recommendations",
            _bullet(recommendations),
        ]
        return "\n".join(lines)

    if response_mode == "visualization":
        if chart_error:
            return chart_error
        lines = [
            "Answer",
            "Generated the requested chart.",
            "",
            "Charts",
            _compact_charts(charts),
        ]
        if insights_text:
            lines.extend(["", "Insights", _bullet(insights_text)])
        return "\n".join(lines)

    lines = [
        "Answer",
        business_summary,
        "",
        "Statistics",
        _compact_statistics(statistics),
        "",
        "Insights",
        _bullet(insights_text or ["No deeper insight was available from the selected analytical engines."]),
    ]
    return "\n".join(lines)


def _quick_answer(
    question: str,
    df: pd.DataFrame,
    profile: dict[str, Any],
    plan: AnalysisPlan,
    results: list[EngineResult],
    fallback: str,
) -> str:
    text = question.lower()

    if _has_any(text, ["list columns", "column names", "what columns"]):
        return "The dataset columns are: " + ", ".join(str(column) for column in df.columns) + "."
    if _has_any(text, ["duplicate", "duplicates", "duplicate rows"]):
        count = int(profile.get("duplicates", {}).get("count", df.duplicated().sum()))
        return f"The dataset contains {_format_count(count, 'duplicate row')}."
    if _has_any(text, ["missing", "null", "nulls", "missing values"]):
        return _missing_values_answer(df, profile)
    if _has_any(text, ["row count", "rows", "how many rows", "number of rows"]):
        return f"The dataset contains {_format_count(len(df), 'row')}."
    if _has_any(text, ["column count", "how many columns", "number of columns"]):
        return f"The dataset contains {_format_count(len(df.columns), 'column')}."
    if _has_any(text, ["datatype", "data type", "dtype", "dtypes"]):
        return _datatype_answer(question, df, plan)
    if _has_any(text, ["unique values", "distinct values", "unique"]):
        return _unique_values_answer(question, df, plan)
    if _has_any(text, ["average", "avg", "mean"]):
        return _aggregate_answer("average", question, df, plan)
    if _has_any(text, ["minimum", "min", "lowest", "smallest"]):
        return _aggregate_answer("minimum", question, df, plan)
    if _has_any(text, ["maximum", "max", "highest", "largest"]):
        return _aggregate_answer("maximum", question, df, plan)
    if _has_any(text, ["sum", "total"]):
        return _aggregate_answer("sum", question, df, plan)

    for result in results:
        if result.insights:
            return result.insights[0]
    return fallback


def _missing_values_answer(df: pd.DataFrame, profile: dict[str, Any]) -> str:
    missing = profile.get("missing_values")
    if not missing:
        missing_counts = df.isna().sum()
        total = int(missing_counts.sum())
        columns_with_missing = int((missing_counts > 0).sum())
        return f"The dataset contains {_format_count(total, 'missing value')} across {_format_count(columns_with_missing, 'column')}."

    total = sum(int(meta.get("count", 0)) for meta in missing.values() if isinstance(meta, dict))
    columns_with_missing = sum(1 for meta in missing.values() if isinstance(meta, dict) and int(meta.get("count", 0)) > 0)
    return f"The dataset contains {_format_count(total, 'missing value')} across {_format_count(columns_with_missing, 'column')}."


def _datatype_answer(question: str, df: pd.DataFrame, plan: AnalysisPlan) -> str:
    column = _referenced_column(question, df, plan)
    if column:
        return f"The datatype of {column} is {df[column].dtype}."
    pairs = [f"{column}: {df[column].dtype}" for column in df.columns]
    return "The column datatypes are: " + "; ".join(pairs) + "."


def _unique_values_answer(question: str, df: pd.DataFrame, plan: AnalysisPlan) -> str:
    column = _referenced_column(question, df, plan)
    if not column:
        return "Please specify a column to list unique values."

    values = df[column].dropna().unique().tolist()
    preview = ", ".join(str(value) for value in values[:20])
    suffix = f" Showing the first 20: {preview}." if len(values) > 20 else f" They are: {preview}."
    return f"{column} has {_format_number(len(values))} unique values.{suffix}"


def _aggregate_answer(kind: str, question: str, df: pd.DataFrame, plan: AnalysisPlan) -> str:
    column = _referenced_column(question, df, plan, numeric_only=True)
    if not column:
        return f"Please specify a numeric column for the {kind}."

    values = pd.to_numeric(df[column], errors="coerce")
    if kind == "average":
        value = values.mean()
    elif kind == "minimum":
        value = values.min()
    elif kind == "maximum":
        value = values.max()
    else:
        value = values.sum()

    return f"The {kind} of {column} is {_format_number(_round_number(value))}."


def _referenced_column(
    question: str,
    df: pd.DataFrame,
    plan: AnalysisPlan,
    numeric_only: bool = False,
) -> str | None:
    text = question.lower()
    for column in df.columns:
        column_text = str(column).lower()
        spaced = column_text.replace("_", " ")
        if re.search(rf"\b{re.escape(column_text)}\b", text) or re.search(rf"\b{re.escape(spaced)}\b", text):
            if not numeric_only or pd.api.types.is_numeric_dtype(df[column]):
                return str(column)

    for match in plan.column_matches.values():
        if match.column and match.column in df.columns:
            if not numeric_only or pd.api.types.is_numeric_dtype(df[match.column]):
                return match.column

    if numeric_only:
        numeric_columns = [str(column) for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
        return numeric_columns[0] if len(numeric_columns) == 1 else None
    return None


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _format_count(value: int, singular: str) -> str:
    plural = singular if singular.endswith("s") else f"{singular}s"
    return f"{_format_number(value)} {singular if value == 1 else plural}"


def _round_number(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
        return round(float(value), 4)
    except Exception:
        return value


def _compact_statistics(statistics: dict[str, Any]) -> str:
    if not statistics:
        return "No numeric statistics were required for this question."
    parts = []
    for engine, metrics in statistics.items():
        if engine == "statistics" and {"matched_count", "total_count", "percentage"}.issubset(metrics):
            filter_meta = metrics.get("filter", {})
            parts.append(
                "- Count: "
                f"{metrics['matched_count']} of {metrics['total_count']} records "
                f"({metrics['percentage']}%) match "
                f"{filter_meta.get('column', 'the selected column')} = {filter_meta.get('value', 'selected value')}."
            )
            continue

        if isinstance(metrics, dict):
            readable = []
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        readable.append(f"{nested_key}: {nested_value}")
                else:
                    readable.append(f"{key}: {value}")
            parts.append(f"- {engine}: " + "; ".join(readable[:10]))
        else:
            parts.append(f"- {engine}: {metrics}")
    return "\n".join(parts)


def _compact_charts(charts: list[dict[str, Any]]) -> str:
    if not charts:
        return "No chart is necessary for this answer."
    return "\n".join(
        f"- {chart['chart_type']}: {chart['title']}"
        + (f" ({chart['image_url']})" if chart.get("image_url") else "")
        for chart in charts
    )


def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _chart_suggestions(plan: AnalysisPlan, results: list[EngineResult]) -> list[dict[str, Any]]:
    if not plan.chart_type:
        return []

    if any(
        result.name == "statistics" and
        isinstance(result.metrics, dict) and
        {"matched_count", "total_count", "percentage"}.issubset(result.metrics)
        for result in results
    ):
        return []

    result_names = {result.name for result in results}
    if plan.chart_type == "pie":
        return [{"chart_type": "pie", "title": "Category share", "reason": "Pie charts show the share of each category."}]
    if plan.chart_type == "histogram":
        return [{"chart_type": "histogram", "title": "Distribution", "reason": "Histograms show the spread and frequency of numeric values."}]
    if plan.chart_type == "line" and "trend" in result_names:
        return [{"chart_type": "line", "title": "Trend over time", "reason": "Time-based movement is easier to interpret visually."}]
    if plan.chart_type == "scatter" and "correlation" in result_names:
        return [{"chart_type": "scatter", "title": "Relationship between numeric variables", "reason": "Scatter plots reveal correlation strength and outliers."}]
    if plan.chart_type == "box" and "outlier" in result_names:
        return [{"chart_type": "box", "title": "Outlier distribution", "reason": "Box plots highlight values outside normal operating range."}]
    if plan.chart_type == "bar":
        return [{"chart_type": "bar", "title": "Business comparison", "reason": "Bar charts make rankings and category comparisons easy to scan."}]
    return []


def _recommendations(plan: AnalysisPlan, results: list[EngineResult], profile: dict[str, Any]) -> list[str]:
    recommendations = []
    failed = [result for result in results if not result.success]
    if plan.clarification:
        recommendations.append("Clarify which business column should be used before making decisions from this answer.")
    if profile.get("duplicates", {}).get("count", 0):
        recommendations.append("Review duplicate rows before publishing executive KPIs.")
    if any(result.name == "outlier" and result.success for result in results):
        recommendations.append("Investigate the highest-impact outlier records before treating them as normal performance.")
    if any(result.name == "trend" and result.success for result in results):
        recommendations.append("Track the trend monthly and compare it with operational events, pricing, promotions or capacity changes.")
    if failed:
        recommendations.append("Add or clean the missing analytical columns so the skipped engines can run reliably.")
    if not recommendations:
        recommendations.append("Use the result as a decision-support signal and validate with business owners before action.")
    return recommendations[:5]


def _confidence(plan: AnalysisPlan, results: list[EngineResult]) -> float:
    match_scores = [match.score for match in plan.column_matches.values()]
    semantic_score = sum(match_scores) / len(match_scores) if match_scores else 0.74
    success_rate = sum(1 for result in results if result.success) / max(len(results), 1)
    penalty = 0.18 if plan.clarification else 0
    return max(0.0, min(1.0, round((semantic_score * 0.55) + (success_rate * 0.45) - penalty, 2)))


def _followups(plan: AnalysisPlan, profile: dict[str, Any]) -> list[str]:
    measures = profile.get("semantic_roles", {}).get("possible_measures", [])
    dimensions = profile.get("semantic_roles", {}).get("possible_dimensions", [])
    measure = measures[0] if measures else "the main KPI"
    dimension = dimensions[0] if dimensions else "business segment"
    return [
        f"What are the top drivers of {measure}?",
        f"Compare {measure} by {dimension}.",
        "Which records or KPIs should management investigate first?",
    ]


def _plan_to_dict(plan: AnalysisPlan) -> dict[str, Any]:
    data = asdict(plan)
    data["column_matches"] = {
        key: asdict(value)
        for key, value in plan.column_matches.items()
    }
    return data


def _json_safe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = []
    for record in records:
        safe.append({
            str(key): _json_safe_value(value)
            for key, value in record.items()
        })
    return safe


def _json_safe_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
