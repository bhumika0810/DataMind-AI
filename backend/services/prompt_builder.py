from __future__ import annotations

import json
from typing import Any


def _safe_json_dumps(value: Any) -> str:
    """Serialize profile data for prompt context without failing on odd values."""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return "{}"


def _as_list(value: Any) -> list[Any]:
    """Return a list only when the value is already list-like for this profile."""
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary only when the value matches the expected profile shape."""
    return value if isinstance(value, dict) else {}


def _format_column_catalog(columns: list[Any]) -> str:
    """Create a compact, readable catalog of available columns."""
    formatted_columns = []

    for column in columns:
        if not isinstance(column, dict):
            continue

        name = column.get("name", "unknown")
        dtype = column.get("dtype", "unknown")
        category = column.get("category", "unknown")
        formatted_columns.append(f"- {name} ({category}, dtype: {dtype})")

    return "\n".join(formatted_columns) if formatted_columns else "- No columns available"


def _build_profile_context(dataset_profile: dict[str, Any]) -> dict[str, Any]:
    """Select the profile fields that are most useful for accurate LLM answers."""
    profile = _as_dict(dataset_profile)

    return {
        "general": _as_dict(profile.get("general")),
        "summary": _as_dict(profile.get("summary")),
        "columns": _as_list(profile.get("columns")),
        "numeric_statistics": _as_dict(profile.get("numeric_statistics")),
        "categorical_statistics": _as_dict(profile.get("categorical_statistics")),
        "datetime_statistics": _as_dict(profile.get("datetime_statistics")),
        "missing_values": _as_dict(profile.get("missing_values")),
        "duplicates": _as_dict(profile.get("duplicates")),
        "preview": _as_list(profile.get("preview")),
    }


def build_dataset_prompt(dataset_profile: dict[str, Any], question: str) -> str:
    """
    Build a structured prompt for answering questions from a dataset profile.

    The prompt is optimized for LLMs that should reason from the profile only,
    avoid unsupported claims, and use exact column names when discussing data.
    """
    context = _build_profile_context(dataset_profile)
    user_question = str(question or "").strip()
    column_catalog = _format_column_catalog(context["columns"])
    profile_json = _safe_json_dumps(context)

    return f"""
You are DataMind AI, an expert data analyst.

Your task is to answer the user's question using only the dataset profile below.
The profile was generated from a pandas DataFrame and may include summary
statistics, missing values, duplicate information, date ranges, frequent values,
and a small preview of rows.

DATASET COLUMNS
{column_catalog}

DATASET PROFILE JSON
{profile_json}

USER QUESTION
{user_question}

RESPONSE RULES
1. Use only the information present in the dataset profile.
2. Do not invent columns, rows, values, statistics, or relationships.
3. Refer to columns by their exact names from DATASET COLUMNS.
4. If the profile does not contain enough information to answer confidently,
   say what is missing and give the best supported answer.
5. Account for missing values and duplicates when they are relevant.
6. For numeric questions, prefer numeric_statistics when available.
7. For category or text questions, prefer categorical_statistics and preview rows.
8. For date or time questions, prefer datetime_statistics and datetime columns.
9. Keep the answer concise, clear, and business-friendly.
10. Do not mention implementation details unless the user asks for them.

Answer the user's question now.
""".strip()
