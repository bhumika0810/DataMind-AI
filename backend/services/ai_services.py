"""
services/ai_service.py

AI Query Engine — the core of DataMind AI.

Flow:
  1. Load the dataset (sample up to MAX_ROWS_FOR_AI rows)
  2. Build a rich prompt: schema + data sample + user question
  3. Send to Gemini
  4. Parse structured response: answer + insights + follow-ups
"""

import json
import re
import pandas as pd
from typing import Any
from services.csv_services import _find_file
from services.llm_services import call_llm
from config import settings
from services.analytics_engine import AnalyticsEngine


# ─────────────────────────────────────────────
#  PUBLIC: answer_question
# ─────────────────────────────────────────────

async def answer_question(
    file_id:  str,
    question: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """
    The main AI query function.
    Returns a structured response with narrative answer, insights, and follow-ups.
    """
    max_rows  = max_rows or settings.MAX_ROWS_FOR_AI
    filepath  = _find_file(file_id)
    df        = pd.read_csv(filepath)
    total_rows = len(df)

    # Smart sampling: prioritise full dataset, fall back to head if too large
    sample_df = _smart_sample(df, max_rows)

    schema_summary = _build_schema_summary(sample_df)
    data_csv       = _df_to_compact_csv(sample_df)

    prompt   = _build_query_prompt(question, schema_summary, data_csv, len(sample_df), total_rows)
    raw      = await call_llm(prompt)
    parsed   = _parse_query_response(raw)

    return {
        "file_id":             file_id,
        "question":            question,
        "answer":              parsed.get("answer", ""),
        "insights":            parsed.get("insights", []),
        "suggested_followups": parsed.get("suggested_followups", []),
        "rows_analyzed":       len(sample_df),
        "model_used":          "llama-3.3-70b-versatile-v0.1-q4_0.gguf",
    }


def answer_question_local(file_id: str, question: str) -> dict[str, Any]:
    """
    Enterprise analytics engine for uploaded datasets.

    This endpoint remains synchronous for the existing frontend contract, but
    it now runs a full intent -> profile -> semantic match -> plan -> execute
    -> insight pipeline instead of shallow chatbot-style matching.
    """
    filepath = _find_file(file_id)
    df = pd.read_csv(filepath)
    return AnalyticsEngine().answer(file_id=file_id, question=question, df=df)


def _answer_extreme_question(df: pd.DataFrame, question_l: str) -> str | None:
    wants_max = any(term in question_l for term in ["highest", "largest", "maximum", "max", "top"])
    wants_min = any(term in question_l for term in ["lowest", "smallest", "minimum", "min", "least"])
    if not wants_max and not wants_min:
        return None

    numeric_col = _select_numeric_column(df, question_l)
    if numeric_col is None:
        return "I could not find a numeric column in the uploaded dataset to calculate that."

    values = pd.to_numeric(df[numeric_col], errors="coerce")
    if values.dropna().empty:
        return f"The column {numeric_col} does not contain numeric values I can rank."

    row_index = values.idxmax() if wants_max else values.idxmin()
    row = df.loc[row_index]
    value = row[numeric_col]
    label_col = _select_label_column(df, numeric_col, question_l)
    label = row[label_col] if label_col else f"row {int(row_index) + 1}"
    direction = "highest" if wants_max else "lowest"

    return f"{label} has the {direction} {numeric_col}: {value}."


def _answer_aggregate_question(df: pd.DataFrame, question_l: str) -> str | None:
    numeric_col = _select_numeric_column(df, question_l)
    if numeric_col is None:
        if "count" in question_l or "how many" in question_l:
            return f"The uploaded dataset contains {len(df)} rows."
        return None

    values = pd.to_numeric(df[numeric_col], errors="coerce").dropna()
    if values.empty:
        return None

    if any(term in question_l for term in ["average", "avg", "mean"]):
        return f"The average {numeric_col} is {values.mean():.2f}."
    if any(term in question_l for term in ["total", "sum"]):
        return f"The total {numeric_col} is {values.sum():.2f}."
    if "count" in question_l or "how many" in question_l:
        return f"The uploaded dataset contains {len(df)} rows."
    return None


def _answer_entity_lookup(df: pd.DataFrame, question: str, question_l: str) -> str | None:
    lookup_terms = _extract_lookup_terms(question, question_l, df)
    if not lookup_terms:
        return None

    matches = _find_matching_rows(df, lookup_terms)
    if matches.empty:
        return f"I could not find records matching {', '.join(lookup_terms)} in the uploaded dataset."

    return _summarize_matching_rows(matches, lookup_terms)


def _extract_lookup_terms(question: str, question_l: str, df: pd.DataFrame) -> list[str]:
    stop_words = {
        "give", "me", "all", "the", "details", "detail", "about", "from", "uploaded",
        "file", "dataset", "data", "customer", "customers", "record", "records", "show",
        "tell", "please", "for", "of", "in", "with", "and", "a", "an"
    }

    text_columns = [
        col for col in df.columns
        if pd.to_numeric(df[col], errors="coerce").dropna().empty
    ]
    if not text_columns:
        return []

    candidates = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", question):
        token_l = token.lower()
        if token_l not in stop_words and len(token_l) > 1:
            candidates.append(token)

    matched_terms = []
    for candidate in candidates:
        candidate_l = candidate.lower()
        for col in text_columns:
            values = df[col].dropna().astype(str).str.lower()
            if values.str.contains(re.escape(candidate_l), na=False).any():
                matched_terms.append(candidate)
                break

    seen = set()
    return [term for term in matched_terms if not (term.lower() in seen or seen.add(term.lower()))]


def _find_matching_rows(df: pd.DataFrame, lookup_terms: list[str]) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    text_columns = [
        col for col in df.columns
        if pd.to_numeric(df[col], errors="coerce").dropna().empty
    ]

    for term in lookup_terms:
        term_mask = pd.Series(False, index=df.index)
        for col in text_columns:
            term_mask = term_mask | df[col].astype(str).str.contains(term, case=False, na=False, regex=False)
        mask = mask | term_mask

    return df[mask].copy()


def _summarize_matching_rows(matches: pd.DataFrame, lookup_terms: list[str]) -> str:
    entity_label = ", ".join(lookup_terms)
    lines = [
        f"I found {len(matches)} matching record{'s' if len(matches) != 1 else ''} for {entity_label}."
    ]

    numeric_columns = [
        col for col in matches.columns
        if not pd.to_numeric(matches[col], errors="coerce").dropna().empty
    ]
    text_columns = [
        col for col in matches.columns
        if col not in numeric_columns
    ]

    order_col = _find_column(matches, ["order id", "order_id", "order", "id"])
    product_col = _find_column(matches, ["product", "item", "sku"])
    status_col = _find_column(matches, ["status", "state"])
    date_col = _find_column(matches, ["order date", "date", "created", "purchased"])
    quantity_col = _find_column(matches, ["quantity", "qty", "units"])
    price_col = _find_column(matches, ["price", "amount", "revenue", "sales", "total"])

    if order_col:
        lines.append(f"Orders: {matches[order_col].nunique(dropna=True)} unique order(s).")

    if product_col:
        products = matches[product_col].dropna().astype(str).unique().tolist()
        lines.append(f"Products: {', '.join(products[:8])}{'...' if len(products) > 8 else ''}.")

    if quantity_col:
        qty = pd.to_numeric(matches[quantity_col], errors="coerce").dropna()
        if not qty.empty:
            lines.append(f"Total quantity: {qty.sum():.0f}.")

    if price_col:
        price = pd.to_numeric(matches[price_col], errors="coerce").dropna()
        if not price.empty:
            lines.append(f"Total {price_col}: {price.sum():.2f}; average {price_col}: {price.mean():.2f}.")

    if status_col:
        status_counts = matches[status_col].dropna().astype(str).value_counts()
        status_summary = ", ".join(f"{status}: {count}" for status, count in status_counts.items())
        if status_summary:
            lines.append(f"Status breakdown: {status_summary}.")

    if date_col:
        dates = pd.to_datetime(matches[date_col], errors="coerce").dropna()
        if not dates.empty:
            lines.append(
                f"Date range: {dates.min().date().isoformat()} to {dates.max().date().isoformat()}."
            )

    preview_cols = _preview_columns(matches, text_columns, numeric_columns)
    preview = matches[preview_cols].head(5).fillna("").to_dict(orient="records")
    if preview:
        lines.append("Sample matching rows:")
        for row in preview:
            row_text = "; ".join(f"{key}: {value}" for key, value in row.items())
            lines.append(f"- {row_text}")

    return "\n".join(lines)


def _find_column(df: pd.DataFrame, preferred_names: list[str]) -> str | None:
    normalized = {str(col).lower().replace("_", " ").strip(): col for col in df.columns}
    for name in preferred_names:
        if name in normalized:
            return normalized[name]
    for name in preferred_names:
        for norm, original in normalized.items():
            if name in norm:
                return original
    return None


def _preview_columns(
    df: pd.DataFrame,
    text_columns: list[str],
    numeric_columns: list[str],
) -> list[str]:
    preferred = []
    for name in ["order id", "customer", "product", "quantity", "price", "order date", "status"]:
        col = _find_column(df, [name])
        if col and col not in preferred:
            preferred.append(col)

    for col in text_columns + numeric_columns:
        if col not in preferred:
            preferred.append(col)
        if len(preferred) >= 7:
            break
    return preferred


def _select_numeric_column(df: pd.DataFrame, question_l: str) -> str | None:
    numeric_columns = [
        col for col in df.columns
        if not pd.to_numeric(df[col], errors="coerce").dropna().empty
    ]
    if not numeric_columns:
        return None

    for col in numeric_columns:
        if str(col).lower() in question_l:
            return col

    question_tokens = set(re.findall(r"[a-z0-9]+", question_l))
    best_col = None
    best_score = 0
    for col in numeric_columns:
        col_tokens = set(re.findall(r"[a-z0-9]+", str(col).lower()))
        score = len(question_tokens & col_tokens)
        if score > best_score:
            best_col = col
            best_score = score

    if best_col is not None:
        return best_col
    return numeric_columns[0]


def _select_label_column(df: pd.DataFrame, numeric_col: str, question_l: str) -> str | None:
    non_numeric_columns = [
        col for col in df.columns
        if col != numeric_col and pd.to_numeric(df[col], errors="coerce").dropna().empty
    ]
    if not non_numeric_columns:
        other_cols = [col for col in df.columns if col != numeric_col]
        return other_cols[0] if other_cols else None

    preferred_names = ["name", "employee", "person", "customer", "product", "user"]
    for preferred in preferred_names:
        for col in non_numeric_columns:
            if preferred in str(col).lower() or preferred in question_l:
                return col
    return non_numeric_columns[0]


# ─────────────────────────────────────────────
#  SMART SAMPLING
# ─────────────────────────────────────────────

def _smart_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """
    Return a representative sample of the dataframe.
    - If df fits in max_rows: return as-is
    - Otherwise: stratified sample (evenly spaced rows to preserve trends)
    """
    if len(df) <= max_rows:
        return df
    step = len(df) // max_rows
    return df.iloc[::step].head(max_rows).reset_index(drop=True)


# ─────────────────────────────────────────────
#  BUILD SCHEMA SUMMARY
# ─────────────────────────────────────────────

def _build_schema_summary(df: pd.DataFrame) -> str:
    """
    Compact schema string for the prompt header.
    """
    lines = [f"Columns ({len(df.columns)} total):"]
    for col in df.columns:
        dtype   = str(df[col].dtype)
        nulls   = int(df[col].isna().sum())
        samples = df[col].dropna().astype(str).head(3).tolist()
        lines.append(f"  - {col} [{dtype}] | nulls: {nulls} | samples: {samples}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  BUILD COMPACT CSV (token-efficient)
# ─────────────────────────────────────────────

def _df_to_compact_csv(df: pd.DataFrame) -> str:
    """
    Convert dataframe to CSV string.
    Numeric floats rounded to 4dp to reduce token count.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include="float").columns:
        df_copy[col] = df_copy[col].round(4)
    return df_copy.to_csv(index=False)


# ─────────────────────────────────────────────
#  BUILD QUERY PROMPT
# ─────────────────────────────────────────────

def _build_query_prompt(
    question:      str,
    schema:        str,
    data_csv:      str,
    sampled_rows:  int,
    total_rows:    int,
) -> str:

    sampling_note = (
        f"(Showing {sampled_rows} of {total_rows} total rows — evenly sampled for analysis)"
        if sampled_rows < total_rows
        else f"(Full dataset: {total_rows} rows)"
    )

    return f"""
You are DataMind AI — an expert AI data analyst embedded in a business intelligence platform.
A user has uploaded a dataset and asked a question. Your job is to analyse the data and answer accurately.

━━━ DATASET SCHEMA ━━━
{schema}

━━━ DATA {sampling_note} ━━━
{data_csv}

━━━ USER QUESTION ━━━
{question}

━━━ YOUR TASK ━━━
Analyse the data above and return a JSON object (ONLY JSON — no markdown, no preamble) with this structure:

{{
  "answer": "<Clear, direct narrative answer to the user's question. 2-5 sentences. Mention specific numbers, names, and trends from the data. Be precise.>",
  "insights": [
    {{
      "type": "text",
      "content": "<Key insight 1 — a specific finding from the data>"
    }},
    {{
      "type": "table",
      "content": {{
        "title": "<Table title>",
        "headers": ["<col1>", "<col2>", "..."],
        "rows": [
          ["<val>", "<val>", "..."],
          ["<val>", "<val>", "..."]
        ]
      }}
    }},
    {{
      "type": "chart_suggestion",
      "content": {{
        "chart_type": "<bar | line | pie | scatter | heatmap>",
        "x_axis": "<column name>",
        "y_axis": "<column name>",
        "title": "<Chart title>",
        "reason": "<Why this chart best visualises the answer>"
      }}
    }}
  ],
  "suggested_followups": [
    "<Follow-up question 1 the user might want to ask next>",
    "<Follow-up question 2>",
    "<Follow-up question 3>"
  ]
}}

Rules:
- The "insights" array should have 2-4 items covering: a key text finding, a data table, and a chart suggestion.
- The table "rows" should contain the top 5-10 most relevant rows (not all data).
- Numbers in the answer must match the actual data — do not hallucinate values.
- suggested_followups must be 3 questions, phrased naturally as a business user would ask.
- Return ONLY valid JSON. No markdown. No explanation outside the JSON.
""".strip()


# ─────────────────────────────────────────────
#  PARSE RESPONSE
# ─────────────────────────────────────────────

def _parse_query_response(raw: str) -> dict[str, Any]:
    """
    Robustly parse Gemini's JSON response.
    Falls back gracefully if parsing fails.
    """
    text = raw.strip()

    # Strip markdown fences if model adds them
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Graceful fallback — return raw text as plain answer
        return {
            "answer":              raw.strip()[:2000],
            "insights":            [],
            "suggested_followups": [],
        }
