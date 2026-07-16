from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import pandas as pd

from services.analytics_engine.composer import compose_response
from services.analytics_engine.engines import execute_plan
from services.analytics_engine.intent import detect_domain, detect_intents
from services.analytics_engine.planner import build_analysis_plan
from services.analytics_engine.profiler import build_enterprise_profile, normalize_dataframe
from services.analytics_engine.semantic import semantic_column_match


class AnalyticsEngine:
    def answer(self, file_id: str, question: str, df: pd.DataFrame) -> dict[str, Any]:
        started = time.perf_counter()

        normalized_df = normalize_dataframe(df)
        profile = build_enterprise_profile(normalized_df)
        columns = [str(column) for column in normalized_df.columns]

        intent = detect_intents(question)
        domain = detect_domain(question, columns)
        column_matches = semantic_column_match(question, normalized_df, profile)
        plan = build_analysis_plan(question, [intent], domain, column_matches)
        results = execute_plan(normalized_df, profile, plan)

        execution_time_ms = int((time.perf_counter() - started) * 1000)
        response = compose_response(
            file_id=file_id,
            question=question,
            df=normalized_df,
            profile=profile,
            plan=plan,
            results=results,
            execution_time_ms=execution_time_ms,
        )

        payload = asdict(response)
        payload.update({
            "file_id": file_id,
            "question": question,
            "rows_analyzed": int(len(normalized_df)),
            "model_used": "datamind-analytics-engine-v1",
        })
        return _json_safe(payload)


def _json_safe(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
