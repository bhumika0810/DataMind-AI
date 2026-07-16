from __future__ import annotations

import json
from typing import Any


def _as_dict(value: Any) -> dict:
    """Return a dictionary when the profile field has the expected shape."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    """Return a list when the profile field has the expected shape."""
    return value if isinstance(value, list) else []


def _safe_json(value: Any) -> str:
    """Serialize profile sections for prompt context without raising errors."""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return "{}"


def _get_columns(dataset_profile: dict) -> list:
    """Extract the profiler column catalog."""
    return _as_list(_as_dict(dataset_profile).get("columns"))


def _exact_column_names(columns: list) -> list[str]:
    """Return exact column names that generated pandas code is allowed to use."""
    names = []
    for column in columns:
        if isinstance(column, dict) and column.get("name") is not None:
            names.append(str(column["name"]))
    return names


def _format_available_columns(columns: list) -> str:
    """Format column name, category, and dtype for the LLM."""
    lines = []
    for column in columns:
        if not isinstance(column, dict):
            continue

        name = column.get("name", "unknown")
        category = column.get("category", "unknown")
        dtype = column.get("dtype", "unknown")
        lines.append(f"- {name} | category: {category} | dtype: {dtype}")

    return "\n".join(lines) if lines else "- No columns available"


def _profile_context(dataset_profile: dict) -> dict:
    """Select only the fields needed for accurate pandas code generation."""
    profile = _as_dict(dataset_profile)
    general = _as_dict(profile.get("general"))
    duplicates = _as_dict(profile.get("duplicates"))

    return {
        "dataset_overview": {
            "rows": general.get("rows", 0),
            "columns": general.get("columns", 0),
        },
        "available_columns": _as_list(profile.get("columns")),
        "numeric_statistics": _as_dict(profile.get("numeric_statistics")),
        "categorical_information": _as_dict(profile.get("categorical_statistics")),
        "datetime_information": _as_dict(profile.get("datetime_statistics")),
        "missing_values": _as_dict(profile.get("missing_values")),
        "duplicate_information": {
            "count": duplicates.get("count", 0),
            "percentage": duplicates.get("percentage", 0.0),
        },
        "preview": _as_list(profile.get("preview")),
    }


def _few_shot_examples() -> str:
    """Examples that teach the LLM to map simple questions to a single pandas expression."""
    return """
Question:
How many rows are in the dataset?
Expected pandas code:
result = len(df)

Question:
How many columns are in the dataset?
Expected pandas code:
result = len(df.columns)

Question:
What are the column names?
Expected pandas code:
result = df.columns.to_series()

Question:
Show the first 5 rows
Expected pandas code:
result = df.head(5)

Question:
Show missing values by column
Expected pandas code:
result = df.isnull().sum()

Question:
Show rows with missing salary
Expected pandas code:
result = df[df["salary"].isnull()]

Question:
How many duplicate rows are there?
Expected pandas code:
result = df.duplicated().sum()

Question:
Show duplicate rows
Expected pandas code:
result = df[df.duplicated()]

Question:
Top 5 employees by salary
Expected pandas code:
result = df.nlargest(5, "salary")

Question:
Bottom 5 employees by salary
Expected pandas code:
result = df.nsmallest(5, "salary")

Question:
Who has the highest salary?
Expected pandas code:
result = df.nlargest(1, "salary")

Question:
Average salary
Expected pandas code:
result = df["salary"].mean()

Question:
Total revenue
Expected pandas code:
result = df["revenue"].sum()

Question:
Median salary
Expected pandas code:
result = df["salary"].median()

Question:
Standard deviation of sales
Expected pandas code:
result = df["sales"].std()

Question:
Employees by department
Expected pandas code:
result = df.groupby("department").size().reset_index(name="count")

Question:
Average salary by department
Expected pandas code:
result = df.groupby("department")["salary"].mean().sort_values(ascending=False)

Question:
Sort employees by salary descending
Expected pandas code:
result = df.sort_values("salary", ascending=False)

Question:
Employees from IT department with salary above 50000
Expected pandas code:
result = df[(df["department"].astype(str).str.contains("it", case=False, na=False)) & (df["salary"] > 50000)]

Question:
Sales after 2022
Expected pandas code:
result = df[df["date"].dt.year > 2022]

Question:
Unique departments
Expected pandas code:
result = df["department"].dropna().unique()

Question:
Value counts for city
Expected pandas code:
result = df["city"].value_counts()

Question:
Monthly sales trend
Expected pandas code:
result = df.groupby(df["date"].dt.to_period("M"))["sales"].sum()

Question:
Search employees in Finance department
Expected pandas code:
result = df[df["department"].astype(str).str.contains("Finance", case=False, na=False)]
""".strip()


def _advanced_few_shot_examples() -> str:
    """Examples that teach the LLM to produce multi-step analytical code ending in `result`."""
    return """
Question:
Which factors contribute the most to revenue?
Expected pandas code:
numeric_df = df.select_dtypes(include="number")
correlations = numeric_df.corr(numeric_only=True)["revenue"].drop("revenue").sort_values(ascending=False)
result = correlations

Question:
Compare this quarter with the previous quarter
Expected pandas code:
quarterly = df.groupby(df["date"].dt.to_period("Q"))["revenue"].sum().sort_index()
comparison = quarterly.tail(2)
change_pct = comparison.pct_change().iloc[-1] * 100
result = {
    "quarterly_totals": comparison,
    "change_percent": change_pct
}

Question:
Which customers are most likely to churn?
Expected pandas code:
last_purchase = df.groupby("customer_id")["date"].max()
days_inactive = (df["date"].max() - last_purchase).dt.days
churn_risk = days_inactive.sort_values(ascending=False)
result = churn_risk.head(10)

Question:
Which products should be discontinued?
Expected pandas code:
product_perf = df.groupby("product").agg(
    total_sales=("sales", "sum"),
    total_units=("quantity", "sum"),
    avg_margin=("margin", "mean")
)
worst_performers = product_perf.sort_values("total_sales").head(10)
result = worst_performers

Question:
Which branches are underperforming?
Expected pandas code:
branch_perf = df.groupby("branch")["revenue"].sum()
avg_perf = branch_perf.mean()
underperforming = branch_perf[branch_perf < avg_perf].sort_values()
result = underperforming

Question:
Find hidden trends that are not obvious
Expected pandas code:
numeric_df = df.select_dtypes(include="number")
corr_matrix = numeric_df.corr(numeric_only=True)
strong_pairs = corr_matrix.unstack().sort_values(ascending=False)
strong_pairs = strong_pairs[strong_pairs < 1.0]
result = strong_pairs.head(10)

Question:
Which employees are statistical outliers?
Expected pandas code:
mean_val = df["salary"].mean()
std_val = df["salary"].std()
z_scores = (df["salary"] - mean_val) / std_val
outliers = df[z_scores.abs() > 2]
result = outliers

Question:
What are the top five business risks shown by this dataset?
Expected pandas code:
missing_pct = df.isnull().mean().mul(100).sort_values(ascending=False)
dup_pct = df.duplicated().mean() * 100
numeric_df = df.select_dtypes(include="number")
high_variance = numeric_df.std().sort_values(ascending=False)
result = {
    "top_missing_columns": missing_pct.head(5),
    "duplicate_row_percentage": dup_pct,
    "highest_variance_columns": high_variance.head(5)
}

Question:
If you were the CEO, what would you do based on this data?
Expected pandas code:
numeric_df = df.select_dtypes(include="number")
top_correlations = numeric_df.corr(numeric_only=True).unstack().sort_values(ascending=False)
top_correlations = top_correlations[top_correlations < 1.0]
category_col = df.select_dtypes(include="object").columns[0] if len(df.select_dtypes(include="object").columns) > 0 else None
segment_summary = df.groupby(category_col).mean(numeric_only=True) if category_col else None
result = {
    "strongest_relationships": top_correlations.head(5),
    "segment_summary": segment_summary,
    "missing_data_hotspots": df.isnull().mean().mul(100).sort_values(ascending=False).head(5)
}

Question:
Generate a complete business report with statistics, charts, key findings, risks, and recommendations
Expected pandas code:
numeric_df = df.select_dtypes(include="number")
summary_stats = numeric_df.describe()
missing_summary = df.isnull().mean().mul(100).sort_values(ascending=False)
duplicate_summary = df.duplicated().sum()
correlations = numeric_df.corr(numeric_only=True).unstack().sort_values(ascending=False)
correlations = correlations[correlations < 1.0]
result = {
    "summary_statistics": summary_stats,
    "missing_value_percentage": missing_summary,
    "duplicate_row_count": duplicate_summary,
    "top_correlations": correlations.head(10)
}
""".strip()


def build_pandas_prompt(dataset_profile: dict, question: str) -> str:
    """
    Build one prompt string for an LLM to generate executable pandas code.

    The returned prompt is not meant to answer the question directly. It is meant
    to force the LLM to return only pandas/python code that operates on df and
    assigns its final answer to a variable named `result`.
    """
    profile = _as_dict(dataset_profile)
    context = _profile_context(profile)
    columns = _get_columns(profile)
    exact_columns = _exact_column_names(columns)
    user_question = str(question or "").strip()

    return f"""
You are the pandas code generator for DataMind AI.

Your ONLY task is to convert the user's natural-language question into executable pandas code.
You are NOT answering the question in English.

DATASET OVERVIEW
Rows: {context["dataset_overview"]["rows"]}
Columns: {context["dataset_overview"]["columns"]}

AVAILABLE COLUMNS
{_format_available_columns(columns)}

EXACT COLUMN NAMES YOU MAY USE
{_safe_json(exact_columns)}

NUMERIC STATISTICS
{_safe_json(context["numeric_statistics"])}

CATEGORICAL INFORMATION
{_safe_json(context["categorical_information"])}

DATETIME INFORMATION
{_safe_json(context["datetime_information"])}

MISSING VALUES
{_safe_json(context["missing_values"])}

DUPLICATE INFORMATION
{_safe_json(context["duplicate_information"])}

PREVIEW ROWS
{_safe_json(context["preview"])}

CODE GENERATION RULES
1. Generate ONLY executable Python/pandas code operating on the variable df.
2. Never import pandas, numpy, or any external library. Use only df and built-in pandas/Series/DataFrame methods.
3. Never create a new DataFrame from scratch or hardcode data values.
4. Never use markdown, backticks, or the string python.
5. Never explain the answer in English sentences.
6. Simple lookups (count, single column, single filter, single sort) MUST be exactly one line:
   result = <single expression>
7. Analytical, comparative, diagnostic, or report-style questions MAY use multiple
   intermediate variables, but the LAST line MUST assign the final answer to a
   variable named result. Never print, never return early, never leave a dangling
   expression with no assignment.
8. result may be a scalar, Series, DataFrame, or a plain Python dict whose values are
   scalars, Series, or DataFrames (used to bundle multiple related findings together,
   e.g. a business report with several sections).
9. Never use eval, exec, __import__, open, os, sys, or any file/network/system access.
10. Only use columns from EXACT COLUMN NAMES. If the user references something with no
    close match at all, set result = df.head(0).

COLUMN MATCHING RULES
The user usually does NOT know the exact column names. Before writing code:
- Identify the business intent (HR, Sales, Finance, Retail, Healthcare, etc.).
- Map the user's wording to the closest matching column(s) using column names,
  categories, dtypes, and preview rows.
- Treat common synonyms as equivalent, e.g.:
  Employee = Staff = Associate
  Department = Dept = Business Unit
  Revenue = Sales = Amount
  Customer = Client
  Location = City = Branch
  Status = Order Status = Delivery Status
- Ignore case, spacing, underscores, and singular/plural differences.
- Example: "Pending Orders" against a column "Order Status" becomes:
  result = df[df["Order Status"] == "Pending"]

ANALYTICAL REASONING RULES (for correlation, comparison, outlier, trend, or
report-style questions)
- "Which factors contribute most to X" -> compute numeric correlations against
  the target column with df.select_dtypes(include="number").corr(numeric_only=True).
- "Compare period A with period B" -> group by the relevant time period
  (dt.to_period, dt.year, dt.month) and use .pct_change() or a difference for
  the comparison.
- "Most likely to churn / inactive / at risk" -> use recency, e.g. the most
  recent date per group and how far it is from the dataset's max date.
- "Should be discontinued / underperforming" -> aggregate the relevant metric per
  group and isolate the lowest performers relative to the mean or a sorted tail.
- "Statistical outliers" -> compute a z-score ((x - mean) / std) or IQR bounds on
  the relevant numeric column and filter rows that exceed the threshold.
- "Hidden trends" -> inspect the full correlation matrix (not just one column),
  unstack it, drop self-correlations (value == 1.0), and surface the strongest
  remaining pairs.
- "Risks" -> combine missing-value percentages, duplicate percentage, and
  high-variance numeric columns into one result dict.
- "If you were the CEO" / "complete business report" -> bundle several of the
  above analyses (correlations, group summaries, missing data, outliers) into
  one result dict with clearly named keys so each section can be rendered
  separately downstream.
- Always reason in this order before writing code:
  1. What is the user asking?
  2. What business domain is this?
  3. Which columns are needed?
  4. Are there filters, groupings, or comparisons involved?
  5. Is this a single-line lookup or a multi-step analysis?
  Only after this, write the code.

FEW-SHOT EXAMPLES (simple lookups)
{_few_shot_examples()}

FEW-SHOT EXAMPLES (advanced analytical and report-style questions)
{_advanced_few_shot_examples()}

USER QUESTION
{user_question}

Return ONLY the Python/pandas code. The last line must assign to `result`.
""".strip()