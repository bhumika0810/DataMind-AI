from __future__ import annotations

import re

from services.analytics_engine.models import AnalysisPlan, ColumnMatch


def build_analysis_plan(
    question: str,
    intents: list[str],
    domain: str,
    column_matches: dict[str, ColumnMatch],
) -> AnalysisPlan:
    intent = intents[0] if intents else "quick_answer"
    question_l = question.lower()
    engines = []
    steps = [
        "Understand business intent and domain context",
        "Profile dataset structure, quality, dates, measures and dimensions",
        "Resolve requested business concepts to dataset columns using semantic confidence scoring",
    ]

    if intent == "quick_answer":
        engines.append("statistics")
        if _has_any(question_l, ["missing", "null", "duplicate", "quality"]):
            engines.append("data_quality")
        steps.append("Compute the requested factual answer without generating report sections")
    elif intent == "analysis":
        if _has_any(question_l, ["trend", "over time", "growth", "decline", "increase", "decrease"]):
            engines.append("trend")
            steps.append("Build time-based movement and growth analysis")
        elif _has_any(question_l, ["correlation", "correlate", "relationship"]):
            engines.append("correlation")
            steps.append("Measure relationships between numeric variables and identify likely drivers")
        elif _has_any(question_l, ["distribution"]):
            engines.append("outlier")
            engines.append("statistics")
            steps.append("Describe distribution and potential outliers")
        else:
            engines.append("statistics")
            steps.append("Compute requested analysis without generating executive report sections")
    elif intent == "visualization":
        if _has_any(question_l, ["trend", "over time", "growth", "decline", "increase", "decrease"]):
            engines.append("trend")
        elif _has_any(question_l, ["correlation", "correlate", "relationship"]):
            engines.append("correlation")
        else:
            engines.append("statistics")
        steps.append("Prepare chart-focused output for the requested visual")
    elif intent == "recommendation":
        engines.append("statistics")
        engines.append("summarization")
        steps.append("Generate requested recommendations from dataset profile and computed signals")
    elif intent == "executive_report":
        engines.append("statistics")
        engines.append("summarization")
        steps.append("Create business-level executive interpretation")

    if not engines:
        engines.append("summarization")
        steps.append("Summarize the dataset and surface useful business questions")

    chart_type = choose_chart_type(intents, question)
    if chart_type:
        steps.append(f"Recommend a {chart_type} chart if the computed result benefits from visualization")

    low_confidence = [
        match for match in column_matches.values()
        if match.column is None and match.score < 0.42
    ]
    clarification = None
    if low_confidence:
        names = ", ".join(match.requested_concept for match in low_confidence)
        clarification = f"I could not confidently identify the column for: {names}."

    return AnalysisPlan(
        question=question,
        intents=intents,
        domain=domain,
        engines=list(dict.fromkeys(engines)),
        steps=steps,
        column_matches=column_matches,
        chart_type=chart_type,
        clarification=clarification,
    )


def choose_chart_type(intents: list[str], question: str = "") -> str | None:
    intent = intents[0] if intents else "quick_answer"
    if intent not in {"visualization", "executive_report"}:
        return None

    question_l = question.lower()
    if _has_any(question_l, ["pie", "donut", "doughnut"]):
        return "pie"
    if _has_any(question_l, ["histogram", "hist", "distribution"]):
        return "histogram"
    if _has_any(question_l, ["scatter"]):
        return "scatter"
    if _has_any(question_l, ["trend", "over time", "growth", "decline", "increase", "decrease"]):
        return "line"
    if _has_any(question_l, ["correlation", "correlate", "relationship"]):
        return "scatter"
    if _has_any(question_l, ["distribution", "outlier", "anomaly"]):
        return "box"
    return "bar"


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)
