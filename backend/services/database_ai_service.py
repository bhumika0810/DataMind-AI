import json
import re
from typing import Any

from services.database_services import execute_sql
from services.llm_services import call_llm


# -------------------------------------------------------
# Generate SQL
# -------------------------------------------------------

async def ask_database(
    question: str,
    schema: dict,
):

    prompt = f"""
You are DataMind AI's SQL planning layer for an enterprise Business Intelligence platform.

You are NOT a chatbot and you are NOT only translating English to SQL.

Before writing SQL, internally perform this analysis:
1. Detect the business intent: aggregation, filtering, sorting, comparison, trend, KPI, risk, anomaly, summary, recommendation.
2. Understand the likely business domain from table and column names.
3. Identify candidate measures, dimensions, date columns, identifiers and relationships from the schema.
4. Match business concepts semantically. Examples:
   - revenue, sales, income, turnover, gross revenue, net revenue may refer to the same business concept.
   - customer, client, buyer, consumer, account may refer to the same business concept.
   - salary, CTC, compensation, package, gross pay may refer to the same business concept.
5. Use only columns that exist in the schema.
6. If the schema does not support the question, return a safe read-only query that shows the relevant available columns or tables.

Database Schema:

{json.dumps(schema, indent=2, default=str)}

Your job is to return ONE valid MySQL query that can produce verified data for a Senior Business Analyst answer.

Rules:

- Return ONLY SQL.
- Do not explain anything.
- Do not use markdown.
- Use LIMIT 100 unless the user asks for more.
- Never invent table names.
- Never invent column names.
- Use only the tables provided.
- Prefer aggregations and comparisons that directly support executive decisions.
- For trend questions, group by an available date/time period.
- For risk/anomaly questions, surface the most suspicious records or extreme values.

Question:

{question}
"""

    sql = await call_llm(prompt)

    sql = clean_sql(sql)

    allowed = [
        "select",
        "show",
        "describe",
        "explain"
    ]

    first_word = sql.strip().split()[0].lower()

    if first_word not in allowed:
        return {
            "question": question,
            "sql": sql,
            "rows": [],
            "answer": "Only read-only SQL queries are allowed."
        }

    rows = execute_sql(sql)

    intent = detect_response_intent(question)
    explanation = await explain_result(
        question,
        sql,
        rows,
        intent,
    )

    return {
        "question": question,
        "sql": sql,
        "rows": rows,
        "answer": explanation
    }


# -------------------------------------------------------
# Remove markdown
# -------------------------------------------------------

def clean_sql(sql: str):

    sql = sql.strip()

    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")

    return sql.strip()


# -------------------------------------------------------
# Explain SQL Result
# -------------------------------------------------------

async def explain_result(
    question: str,
    sql: str,
    rows,
    intent: str | None = None,
):
    intent = intent or detect_response_intent(question)

    if intent == "factual":
        return format_factual_answer(question, rows)

    if intent != "executive_report":
        return await explain_concise_result(question, sql, rows)

    prompt = f"""
You are DataMind AI, a Senior Business Intelligence Analyst.

A SQL query has already been executed.

User Question:

{question}

Executed SQL:

{sql}

SQL Result:

{json.dumps(rows, indent=2, default=str)}

Generate a board-ready answer with these exact sections:

Answer
Business Summary
Statistics
Charts
Insights
Recommendations
Confidence Score
Execution Time

Rules:

- Explain the result in professional business English.
- If multiple rows exist, summarize the business pattern.
- If the result is empty, clearly say no matching records were found and recommend what to check next.
- Do not mention SQL syntax unless it affects a limitation.
- Recommend a chart when useful: line for trends, bar for comparisons, scatter for relationships, heatmap for matrices, box plot for outliers.
- Include limitations and confidence based on whether the returned data fully answers the question.
"""

    return await call_llm(prompt)


async def explain_concise_result(
    question: str,
    sql: str,
    rows,
):

    prompt = f"""
You are DataMind AI, a concise database analyst.

A SQL query has already been executed.

User Question:

{question}

Executed SQL:

{sql}

SQL Result:

{json.dumps(rows, indent=2, default=str)}

Answer the user's question directly in plain business English.

Rules:

- Keep the answer concise.
- Do not generate an executive report.
- Do not include Business Summary, Recommendations, Confidence Score, or board-ready sections.
- If the result is empty, clearly say no matching records were found.
- Do not mention SQL syntax unless it affects a limitation.
"""

    return await call_llm(prompt)


# -------------------------------------------------------
# Response intent and factual formatting
# -------------------------------------------------------

def detect_response_intent(question: str) -> str:

    text = _normalize(question)

    if _matches_any(text, [
        r"\bexecutive report\b",
        r"\bmanagement summary\b",
        r"\bbusiness report\b",
        r"\bboard report\b",
    ]):
        return "executive_report"

    if _matches_any(text, [
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bcount\b",
        r"\baverage\b",
        r"\bavg\b",
        r"\bmean\b",
        r"\bhighest\b",
        r"\blargest\b",
        r"\bmaximum\b",
        r"\bmax\b",
        r"\blowest\b",
        r"\bsmallest\b",
        r"\bminimum\b",
        r"\bmin\b",
        r"\bsum\b",
        r"\btotal\b",
        r"\blist\b",
        r"\bshow\s+all\b",
        r"\bwhat are the\b",
    ]):
        return "factual"

    return "concise"


def format_factual_answer(question: str, rows) -> str:

    if not rows:
        return "No matching records were found."

    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return "No matching records were found."

    if _is_list_question(question):
        list_answer = _format_list_answer(question, normalized_rows)
        if list_answer:
            return list_answer

    scalar = _first_scalar(normalized_rows)
    if scalar is not None and _single_value_result(normalized_rows):
        return _format_scalar_answer(question, scalar)

    if len(normalized_rows) == 1:
        pairs = [
            f"{_humanize_key(key)} is {_format_value(value)}"
            for key, value in normalized_rows[0].items()
        ]
        return _capitalize_sentence(", ".join(pairs)) + "."

    single_column_values = _single_column_values(normalized_rows)
    if single_column_values is not None:
        values = ", ".join(_format_value(value) for value in single_column_values[:20])
        suffix = " Showing the first 20." if len(single_column_values) > 20 else ""
        return f"The results are: {values}.{suffix}"

    return f"The query returned {len(normalized_rows)} rows."


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _normalize_rows(rows) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(row)
        else:
            normalized.append({str(index + 1): value for index, value in enumerate(row)})
    return normalized


def _is_list_question(question: str) -> bool:
    text = _normalize(question)
    return bool(re.search(r"\b(list|show|what are)\b", text))


def _format_list_answer(question: str, rows: list[dict[str, Any]]) -> str | None:
    values = _single_column_values(rows)
    if values is None:
        return None

    subject = _list_subject(question)
    display_values = ", ".join(_format_value(value) for value in values[:20])
    suffix = " Showing the first 20." if len(values) > 20 else ""
    return f"The {subject} are: {display_values}.{suffix}"


def _list_subject(question: str) -> str:
    text = _normalize(question)
    match = re.search(r"\b(?:list|show|what are)(?:\s+all|\s+the)?\s+([a-z0-9_ ]+?)(?:[?.!]|$)", text)
    if match:
        subject = match.group(1).strip()
        subject = re.sub(r"\b(in|from|by|for|where)\b.*$", "", subject).strip()
        if subject:
            return subject
    return "results"


def _first_scalar(rows: list[dict[str, Any]]) -> Any | None:
    if not rows:
        return None
    first_row = rows[0]
    if not first_row:
        return None
    return next(iter(first_row.values()))


def _single_value_result(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == 1 and len(rows[0]) == 1


def _single_column_values(rows: list[dict[str, Any]]) -> list[Any] | None:
    if not rows:
        return None
    first_keys = list(rows[0].keys())
    if len(first_keys) != 1:
        return None
    key = first_keys[0]
    if any(list(row.keys()) != [key] for row in rows):
        return None
    return [row.get(key) for row in rows]


def _factual_label(question: str) -> str:
    text = _normalize(question)
    if _matches_any(text, [r"\bhow many\b", r"\bnumber of\b", r"\bcount\b"]):
        return "count"
    if _matches_any(text, [r"\baverage\b", r"\bavg\b", r"\bmean\b"]):
        return "average"
    if _matches_any(text, [r"\bhighest\b", r"\blargest\b", r"\bmaximum\b", r"\bmax\b"]):
        return "highest value"
    if _matches_any(text, [r"\blowest\b", r"\bsmallest\b", r"\bminimum\b", r"\bmin\b"]):
        return "lowest value"
    if _matches_any(text, [r"\bsum\b", r"\btotal\b"]):
        return "total"
    return "requested value"


def _format_scalar_answer(question: str, value: Any) -> str:
    text = _normalize(question)
    subject = _question_subject(text)
    formatted = _format_value(value)

    if _matches_any(text, [r"\bhow many\b", r"\bnumber of\b", r"\bcount\b"]):
        if subject:
            return f"There are {formatted} {subject}."
        return f"The count is {formatted}."
    if _matches_any(text, [r"\baverage\b", r"\bavg\b", r"\bmean\b"]):
        return f"The average {subject or 'value'} is {formatted}."
    if _matches_any(text, [r"\bhighest\b", r"\blargest\b", r"\bmaximum\b", r"\bmax\b"]):
        return f"The highest {subject or 'value'} is {formatted}."
    if _matches_any(text, [r"\blowest\b", r"\bsmallest\b", r"\bminimum\b", r"\bmin\b"]):
        return f"The lowest {subject or 'value'} is {formatted}."
    if _matches_any(text, [r"\bsum\b", r"\btotal\b"]):
        return f"The total {subject or 'value'} is {formatted}."
    return f"The requested value is {formatted}."


def _question_subject(text: str) -> str:
    patterns = [
        r"\bhow many\s+([a-z0-9_ ]+?)(?:\?|$)",
        r"\bnumber of\s+([a-z0-9_ ]+?)(?:\?|$)",
        r"\b(?:average|avg|mean|highest|largest|maximum|max|lowest|smallest|minimum|min|sum|total)\s+([a-z0-9_ ]+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            subject = match.group(1).strip()
            subject = re.sub(r"\b(in|from|by|for|where|with)\b.*$", "", subject).strip()
            if subject:
                return subject
    return ""


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _humanize_key(key: str) -> str:
    return str(key).replace("_", " ").strip()


def _capitalize_sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
