from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColumnMatch:
    requested_concept: str
    column: str | None
    score: float
    candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisPlan:
    question: str
    intents: list[str]
    domain: str
    engines: list[str]
    steps: list[str]
    column_matches: dict[str, ColumnMatch]
    chart_type: str | None = None
    clarification: str | None = None


@dataclass
class EngineResult:
    name: str
    success: bool
    generated_code: str
    data: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class AnalyticsResponse:
    answer: str
    business_summary: str
    statistics: dict[str, Any]
    charts: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    recommendations: list[str]
    confidence_score: float | None
    execution_time_ms: int
    analysis_plan: dict[str, Any]
    generated_code: list[str]
    suggested_followups: list[str]
