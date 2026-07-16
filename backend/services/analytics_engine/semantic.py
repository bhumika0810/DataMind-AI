from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from services.analytics_engine.models import ColumnMatch


SEMANTIC_CONCEPTS = {
    "revenue": ["revenue", "sales", "income", "turnover", "business value", "net revenue", "gross revenue", "amount"],
    "salary": ["salary", "annual salary", "monthly salary", "ctc", "compensation", "gross pay", "net pay", "income", "package", "employee cost"],
    "customer": ["customer", "client", "buyer", "consumer", "account"],
    "employee": ["employee", "staff", "worker", "associate", "personnel"],
    "department": ["department", "dept", "function", "team", "division"],
    "product": ["product", "item", "sku", "material", "model"],
    "region": ["region", "zone", "territory", "state", "city", "country", "market"],
    "date": ["date", "time", "period", "month", "quarter", "year", "created", "order date"],
    "cost": ["cost", "expense", "spend", "opex", "capex"],
    "profit": ["profit", "margin", "net income", "earnings"],
    "quantity": ["quantity", "qty", "units", "volume"],
    "status": ["status", "state", "stage", "pending", "completed", "cancelled", "open", "closed"],
    "order": ["order", "order id", "order number", "invoice", "transaction", "booking"],
    "risk": ["risk", "fraud", "anomaly", "failure"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def semantic_column_match(
    question: str,
    df: pd.DataFrame,
    profile: dict[str, Any],
    threshold: float = 0.42,
) -> dict[str, ColumnMatch]:
    question_l = question.lower()
    concepts = _requested_concepts(question_l)
    if not concepts:
        concepts = _infer_default_concepts(question_l, profile)

    return {
        concept: _match_concept_to_column(concept, df, profile, threshold)
        for concept in concepts
    }


def _requested_concepts(question_l: str) -> list[str]:
    concepts = []
    for concept, synonyms in SEMANTIC_CONCEPTS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", question_l) for term in synonyms):
            concepts.append(concept)
    return concepts


def _infer_default_concepts(question_l: str, profile: dict[str, Any]) -> list[str]:
    concepts = []
    if any(term in question_l for term in ["how many", "number of", "count"]):
        concepts.append("status")
        if any(term in question_l for term in ["order", "orders"]):
            concepts.append("order")
    if any(term in question_l for term in ["top", "highest", "lowest", "average", "total", "sum"]):
        concepts.append("revenue")
    if any(term in question_l for term in ["trend", "monthly", "quarter", "year", "growth"]):
        concepts.append("date")
    if not concepts and profile.get("semantic_roles", {}).get("possible_measures"):
        concepts.append("measure")
    return list(dict.fromkeys(concepts))


def _match_concept_to_column(
    concept: str,
    df: pd.DataFrame,
    profile: dict[str, Any],
    threshold: float,
) -> ColumnMatch:
    synonyms = SEMANTIC_CONCEPTS.get(concept, [concept])
    candidates = []

    for column in df.columns:
        score = _score_column(str(column), synonyms)
        category = _column_category(profile, str(column))

        if concept in {"revenue", "salary", "cost", "profit", "quantity", "measure"} and category == "numeric":
            score += 0.18
        if concept == "date" and category == "datetime":
            score += 0.3
        if concept in {"customer", "employee", "department", "product", "region", "status", "order"} and category in {"categorical", "text"}:
            score += 0.12

        candidates.append({"column": str(column), "score": round(min(score, 1.0), 3), "category": category})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0] if candidates else {"column": None, "score": 0}
    selected = best["column"] if best["score"] >= threshold else None

    return ColumnMatch(
        requested_concept=concept,
        column=selected,
        score=float(best["score"]),
        candidates=candidates[:5],
    )


def _score_column(column: str, synonyms: list[str]) -> float:
    column_tokens = set(tokenize(column))
    best = 0.0
    column_l = column.lower().replace("_", " ")

    for synonym in synonyms:
        synonym_tokens = set(tokenize(synonym))
        token_overlap = len(column_tokens & synonym_tokens) / max(len(synonym_tokens), 1)
        sequence = SequenceMatcher(None, column_l, synonym).ratio()
        contains = 1.0 if synonym in column_l or column_l in synonym else 0.0
        best = max(best, token_overlap * 0.65 + sequence * 0.25 + contains * 0.25)

    return min(best, 1.0)


def _column_category(profile: dict[str, Any], column: str) -> str:
    for item in profile.get("columns", []):
        if item.get("name") == column:
            return item.get("category", "text")
    return "text"
