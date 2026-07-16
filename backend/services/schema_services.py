"""
services/schema_service.py

Schema Intelligence — uses Gemini to understand what a dataset means:
  - Describe each column in plain English
  - Identify the business domain (e-commerce, HR, finance, etc.)
  - Write a dataset summary
  - Suggest 5 natural language questions the user could ask
"""

import json
import pandas as pd
from typing import Any
from services.csv_services import _find_file
from services.llm_services import call_llm


# ─────────────────────────────────────────────
#  PUBLIC: ANALYSE SCHEMA
# ─────────────────────────────────────────────

async def analyse_schema(file_id: str) -> dict[str, Any]:
    """
    Load the CSV, build a compact schema snapshot,
    send it to Gemini, and return structured intelligence.
    """
    filepath = _find_file(file_id)
    df       = pd.read_csv(filepath, nrows=50)   # sample for schema understanding

    schema_snapshot = _build_schema_snapshot(df)
    prompt          = _build_schema_prompt(schema_snapshot)
    raw_response    = await call_llm(prompt)
    parsed          = _parse_schema_response(raw_response)

    return {
        "file_id":               file_id,
        "dataset_summary":       parsed.get("dataset_summary", ""),
        "likely_domain":         parsed.get("likely_domain", ""),
        "suggested_questions":   parsed.get("suggested_questions", []),
        "columns":               parsed.get("columns", []),
    }


# ─────────────────────────────────────────────
#  BUILD SCHEMA SNAPSHOT (compact, token-efficient)
# ─────────────────────────────────────────────

def _build_schema_snapshot(df: pd.DataFrame) -> list[dict]:
    """
    For each column, collect dtype, nulls, unique count, and 5 sample values.
    Kept compact to minimise token usage.
    """
    snapshot = []
    for col in df.columns:
        series = df[col]
        entry  = {
            "name":         col,
            "dtype":        str(series.dtype),
            "null_pct":     round(series.isna().mean() * 100, 1),
            "unique_count": int(series.nunique(dropna=True)),
            "samples":      series.dropna().astype(str).head(5).tolist(),
        }
        if pd.api.types.is_numeric_dtype(series):
            entry["min"] = round(float(series.min()), 4)
            entry["max"] = round(float(series.max()), 4)
        snapshot.append(entry)
    return snapshot


# ─────────────────────────────────────────────
#  BUILD PROMPT
# ─────────────────────────────────────────────

def _build_schema_prompt(schema: list[dict]) -> str:
    schema_str = json.dumps(schema, indent=2)
    return f"""
You are a senior data analyst and business intelligence expert.
Below is a schema snapshot of an uploaded dataset (column names, data types, sample values, etc.).

SCHEMA:
{schema_str}

Your job is to analyse this schema and return a JSON object (and ONLY a JSON object — no markdown, no explanation outside JSON) with this exact structure:

{{
  "dataset_summary": "<2-3 sentence plain-English description of what this dataset contains and its likely purpose>",
  "likely_domain": "<one short phrase, e.g. 'E-commerce Sales', 'HR Employee Data', 'Financial Transactions', 'Healthcare Records'>",
  "suggested_questions": [
    "<natural language question 1 a business user would ask>",
    "<natural language question 2>",
    "<natural language question 3>",
    "<natural language question 4>",
    "<natural language question 5>"
  ],
  "columns": [
    {{
      "name": "<exact column name from schema>",
      "ai_description": "<plain English: what this column represents>",
      "business_role": "<one of: metric | dimension | identifier | date | text | unknown>",
      "suggested_label": "<human-friendly display label, e.g. 'Total Revenue'>"
    }}
  ]
}}

Rules:
- Every column in the schema must appear in the "columns" array.
- business_role must be exactly one of: metric, dimension, identifier, date, text, unknown.
- suggested_questions must be 5 items, phrased as a non-technical business user would ask them.
- Return ONLY valid JSON. No markdown fences. No preamble.
""".strip()


# ─────────────────────────────────────────────
#  PARSE GEMINI RESPONSE
# ─────────────────────────────────────────────

def _parse_schema_response(raw: str) -> dict:
    """
    Robustly parse Gemini's JSON response.
    Strips markdown fences if model adds them anyway.
    """
    text = raw.strip()

    # Strip ```json ... ``` if model wrapped it anyway
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Return a safe fallback instead of crashing
        return {
            "dataset_summary":     "Could not parse AI response.",
            "likely_domain":       "Unknown",
            "suggested_questions": [],
            "columns":             [],
            "_raw_error":          str(e),
            "_raw_response":       raw[:500],
        }