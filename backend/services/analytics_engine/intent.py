from __future__ import annotations

import re


SUPPORTED_INTENTS = {
    "quick_answer",
    "analysis",
    "executive_report",
    "visualization",
    "recommendation",
}

FACTUAL_PATTERNS = [
    r"\bhow many\s+(rows?|columns?|records?)\b",
    r"\bnumber of\s+(rows?|columns?|records?)\b",
    r"\b(row|column|record)\s+count\b",
    r"\bcount\s+(rows?|columns?|records?)\b",
    r"\blist\s+(all\s+)?columns?\b",
    r"\b(show|what are|give me)\s+(the\s+)?columns?\b",
    r"\b(column\s+names?|schema)\b",
    r"\b(missing values?|missing data|nulls?|null values?|nan values?)\b",
    r"\b(duplicates?|duplicate rows?)\b",
    r"\b(data ?types?|dtypes?|column types?)\b",
    r"\b(unique|distinct)\s+(values?|count)\b",
    r"\b(average|avg|mean|minimum|min|maximum|max|sum|total|median|mode)\b",
    r"\bwhat\s+is\s+the\s+(average|avg|mean|minimum|min|maximum|max|sum|total|median|mode)\b",
]

EXECUTIVE_REPORT_PATTERNS = [
    r"\bexecutive summary\b",
    r"\bbusiness report\b",
    r"\bboard report\b",
    r"\bmanagement summary\b",
]

VISUALIZATION_PATTERNS = [
    r"\b(chart|charts|graph|graphs|plot|plots)\b",
    r"\b(visualize|visualise|visualization|visualisation)\b",
]

RECOMMENDATION_PATTERNS = [
    r"\b(recommend|recommendation|recommendations)\b",
    r"\b(suggest|suggestion|suggestions)\b",
    r"\b(optimize|optimise|optimization|optimisation)\b",
    r"\b(improve|improvement|improvements)\b",
]

ANALYSIS_PATTERNS = [
    r"\b(analyze|analyse|analysis)\b",
    r"\bexplain\b",
    r"\btrend(s|ing)?\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bcorrelation\b",
    r"\bcorrelate\b",
    r"\bdistribution\b",
    r"\b(summarize|summarise|summary)\b",
]


DOMAIN_KEYWORDS = {
    "HR": ["employee", "salary", "attrition", "department", "hire", "performance", "ctc"],
    "Finance": ["revenue", "profit", "cost", "margin", "expense", "budget", "cash", "invoice"],
    "Sales": ["sales", "customer", "order", "product", "deal", "pipeline", "quota"],
    "Marketing": ["campaign", "lead", "conversion", "channel", "impression", "click"],
    "Manufacturing": ["machine", "production", "defect", "downtime", "plant", "shift"],
    "Operations": ["process", "sla", "ticket", "cycle", "throughput"],
    "Logistics": ["shipment", "delivery", "route", "carrier", "freight"],
    "Inventory": ["inventory", "stock", "warehouse", "reorder", "shortage"],
    "Healthcare": ["patient", "hospital", "diagnosis", "doctor", "claim"],
    "Education": ["student", "course", "grade", "attendance", "faculty"],
    "Retail": ["store", "sku", "basket", "price", "discount"],
    "Supply Chain": ["supplier", "procurement", "lead time", "purchase"],
    "CRM": ["account", "customer", "churn", "retention", "contact"],
    "ERP": ["vendor", "invoice", "purchase order", "asset"],
    "Banking": ["loan", "account", "transaction", "deposit", "credit"],
    "Insurance": ["policy", "premium", "claim", "loss", "underwriting"],
}


def detect_intent(question: str) -> str:
    text = _normalize(question)

    # Factual questions win over every other mode by product contract.
    if _matches_any(text, FACTUAL_PATTERNS):
        return "quick_answer"
    if _matches_any(text, EXECUTIVE_REPORT_PATTERNS):
        return "executive_report"
    if _matches_any(text, VISUALIZATION_PATTERNS):
        return "visualization"
    if _matches_any(text, RECOMMENDATION_PATTERNS):
        return "recommendation"
    if _matches_any(text, ANALYSIS_PATTERNS):
        return "analysis"
    return "quick_answer"


def detect_intents(question: str) -> str:
    return detect_intent(question)


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def detect_domain(question: str, columns: list[str]) -> str:
    haystack = " ".join([question, *columns]).lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", haystack))

    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score else "General Business"
