from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from config import settings
from services.analytics_engine.models import AnalysisPlan


CHART_DIR = Path(settings.DATA_DIR) / "charts"
CHART_URL_PREFIX = "/generated-charts"


def generate_chart(
    question: str,
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "success": False,
            "error": f"Chart generation requires matplotlib: {exc}",
        }

    try:
        chart_type = _requested_chart_type(question, plan)
        figure, axis = plt.subplots(figsize=(9, 5.2), dpi=140)

        if chart_type == "line":
            _plot_line(axis, df)
        elif chart_type == "pie":
            _plot_pie(axis, df)
        elif chart_type == "histogram":
            _plot_histogram(axis, df)
        elif chart_type == "scatter":
            _plot_scatter(axis, df)
        else:
            _plot_bar(axis, df)

        axis.set_title(_chart_title(question, chart_type), fontsize=13, fontweight="bold")
        figure.tight_layout()

        CHART_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.png"
        filepath = CHART_DIR / filename
        figure.savefig(filepath, format="png", bbox_inches="tight")
        plt.close(figure)

        return {
            "success": True,
            "chart_type": chart_type,
            "title": _chart_title(question, chart_type),
            "image_url": f"{CHART_URL_PREFIX}/{filename}",
            "image_path": str(filepath),
        }
    except Exception as exc:
        try:
            plt.close("all")
        except Exception:
            pass
        return {
            "success": False,
            "error": f"Could not generate the requested chart: {exc}",
        }


def _requested_chart_type(question: str, plan: AnalysisPlan) -> str:
    text = question.lower()
    if any(term in text for term in ["pie", "donut", "doughnut"]):
        return "pie"
    if any(term in text for term in ["histogram", "hist", "distribution"]):
        return "histogram"
    if "scatter" in text:
        return "scatter"
    if any(term in text for term in ["line", "trend", "over time", "time series"]):
        return "line"
    if any(term in text for term in ["bar", "column chart"]):
        return "bar"
    if plan.chart_type in {"line", "scatter", "pie", "histogram", "bar"}:
        return plan.chart_type
    return "bar"


def _plot_bar(axis: Any, df: pd.DataFrame) -> None:
    numeric = _numeric_columns(df)
    categorical = _categorical_columns(df)

    if categorical and numeric:
        dimension = categorical[0]
        measure = numeric[0]
        series = (
            df.groupby(dimension, dropna=False)[measure]
            .sum()
            .sort_values(ascending=False)
            .head(12)
        )
        labels = [str(item) for item in series.index]
        axis.bar(labels, series.values, color="#2563eb")
        axis.set_xlabel(dimension)
        axis.set_ylabel(measure)
        axis.tick_params(axis="x", labelrotation=35)
        return

    if numeric:
        measure = numeric[0]
        values = pd.to_numeric(df[measure], errors="coerce").dropna().head(20)
        axis.bar(range(1, len(values) + 1), values.values, color="#2563eb")
        axis.set_xlabel("Row")
        axis.set_ylabel(measure)
        return

    counts = df.iloc[:, 0].astype(str).value_counts().head(12)
    axis.bar(counts.index.astype(str), counts.values, color="#2563eb")
    axis.set_xlabel(str(df.columns[0]))
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", labelrotation=35)


def _plot_line(axis: Any, df: pd.DataFrame) -> None:
    numeric = _numeric_columns(df)
    if not numeric:
        raise ValueError("a line chart requires at least one numeric column")

    measure = numeric[0]
    date_col = _date_column(df)
    if date_col:
        working = df[[date_col, measure]].copy()
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working[measure] = pd.to_numeric(working[measure], errors="coerce")
        working = working.dropna()
        if working.empty:
            raise ValueError("no valid date and numeric rows were available")
        series = working.set_index(date_col).sort_index()[measure].resample("ME").sum()
        axis.plot(series.index.astype(str), series.values, marker="o", color="#2563eb")
        axis.set_xlabel(date_col)
        axis.set_ylabel(measure)
        axis.tick_params(axis="x", labelrotation=35)
        return

    values = pd.to_numeric(df[measure], errors="coerce").dropna().head(50)
    if values.empty:
        raise ValueError("no numeric values were available")
    axis.plot(range(1, len(values) + 1), values.values, marker="o", color="#2563eb")
    axis.set_xlabel("Row")
    axis.set_ylabel(measure)


def _plot_pie(axis: Any, df: pd.DataFrame) -> None:
    categorical = _categorical_columns(df)
    if not categorical:
        raise ValueError("a pie chart requires a categorical column")

    column = categorical[0]
    counts = df[column].astype(str).value_counts().head(8)
    if counts.empty:
        raise ValueError("no category values were available")
    axis.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
    axis.axis("equal")


def _plot_histogram(axis: Any, df: pd.DataFrame) -> None:
    numeric = _numeric_columns(df)
    if not numeric:
        raise ValueError("a histogram requires a numeric column")

    measure = numeric[0]
    values = pd.to_numeric(df[measure], errors="coerce").dropna()
    if values.empty:
        raise ValueError("no numeric values were available")
    axis.hist(values.values, bins=min(20, max(5, int(len(values) ** 0.5))), color="#2563eb", edgecolor="white")
    axis.set_xlabel(measure)
    axis.set_ylabel("Frequency")


def _plot_scatter(axis: Any, df: pd.DataFrame) -> None:
    numeric = _numeric_columns(df)
    if len(numeric) < 2:
        raise ValueError("a scatter plot requires at least two numeric columns")

    x_col, y_col = numeric[:2]
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    working = pd.DataFrame({x_col: x_values, y_col: y_values}).dropna()
    if working.empty:
        raise ValueError("no paired numeric values were available")
    axis.scatter(working[x_col], working[y_col], color="#2563eb", alpha=0.75)
    axis.set_xlabel(x_col)
    axis.set_ylabel(y_col)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        columns.append(str(column))
    return columns


def _date_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return str(column)
    for column in df.columns:
        if any(term in str(column).lower() for term in ["date", "time", "month", "year"]):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().sum() >= max(1, int(df[column].notna().sum() * 0.7)):
                return str(column)
    return None


def _chart_title(question: str, chart_type: str) -> str:
    cleaned = " ".join(str(question or "").strip().split())
    if cleaned:
        return cleaned[:90]
    return f"{chart_type.title()} Chart"
