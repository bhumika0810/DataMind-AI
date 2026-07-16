"""
routes/ai_routes.py

AI Query Endpoints:

  POST /api/ai/query        — Ask a natural language question about a dataset
  POST /api/ai/summarise    — Auto-generate an executive summary of a dataset
  GET  /api/ai/suggestions  — Get AI-generated question suggestions for a dataset
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services import ai_services, schema_services
from schemas import QueryRequest, QueryResponse

logger = logging.getLogger("datamind.ai_routes")
router = APIRouter()


# ─────────────────────────────────────────────
#  POST /query   ← THE CORE ENDPOINT
# ─────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a natural language question about your dataset",
    description="""
The heart of DataMind AI. Upload a dataset, then ask anything:

- "Which product had the highest revenue last quarter?"
- "Show me the top 10 customers by order value"
- "What is the average salary by department?"
- "Which region had the fastest growth?"

Returns a structured response with:
- **answer**: narrative explanation
- **insights**: data tables + chart suggestions
- **suggested_followups**: next questions to ask
    """,
)
async def query_dataset(request: QueryRequest):
    try:
        result = await ai_services.answer_question(
            file_id=request.file_id,
            question=request.question,
            max_rows=request.max_rows,
        )
        return QueryResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # Gemini API errors
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")
    except Exception as e:
        logger.error(f"Query failed for file {request.file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.post(
    "/query-local",
    response_model=QueryResponse,
    summary="Ask a deterministic analytics question about your dataset",
)
def query_dataset_local(request: QueryRequest):
    try:
        result = ai_services.answer_question_local(
            file_id=request.file_id,
            question=request.question,
        )
        return QueryResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Local query failed for file {request.file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ─────────────────────────────────────────────
#  POST /summarise
# ─────────────────────────────────────────────

class SummariseRequest(BaseModel):
    file_id: str = Field(..., description="UUID of the uploaded dataset")


class SummariseResponse(BaseModel):
    file_id:          str
    executive_summary: str
    key_metrics:      list[str]
    data_quality_notes: list[str]
    recommended_analyses: list[str]


@router.post(
    "/summarise",
    response_model=SummariseResponse,
    summary="Generate an executive summary of a dataset",
    description="Auto-generates a business-ready summary, key metrics found, data quality notes, and recommended analyses.",
)
async def summarise_dataset(request: SummariseRequest):
    # We re-use the query engine with a structured summarisation question
    summarise_question = (
        "Please provide a comprehensive executive summary of this dataset. "
        "Include: what the data contains, the time range if applicable, "
        "key metrics and their ranges, any notable trends, and data quality observations."
    )

    try:
        result = await ai_services.answer_question(
            file_id=request.file_id,
            question=summarise_question,
            max_rows=settings_max(),
        )

        # Also get schema for metadata
        schema = await schema_services.analyse_schema(request.file_id)

        return SummariseResponse(
            file_id=request.file_id,
            executive_summary=result["answer"],
            key_metrics=_extract_key_metrics(result["insights"]),
            data_quality_notes=_extract_quality_notes(schema),
            recommended_analyses=result.get("suggested_followups", []),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")
    except Exception as e:
        logger.error(f"Summarise failed for {request.file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  GET /suggestions/{file_id}
# ─────────────────────────────────────────────

class SuggestionsResponse(BaseModel):
    file_id:   str
    questions: list[str]
    domain:    str


@router.get(
    "/suggestions/{file_id}",
    response_model=SuggestionsResponse,
    summary="Get AI-suggested questions for a dataset",
    description="Returns 5 smart questions the AI thinks are worth asking about this specific dataset.",
)
async def get_suggestions(file_id: str):
    try:
        schema = await schema_services.analyse_schema(file_id)
        return SuggestionsResponse(
            file_id=file_id,
            questions=schema.get("suggested_questions", []),
            domain=schema.get("likely_domain", "Unknown"),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")
    except Exception as e:
        logger.error(f"Suggestions failed for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def settings_max():
    from config import settings
    return settings.MAX_ROWS_FOR_AI


def _extract_key_metrics(insights: list) -> list[str]:
    """Pull text insights out of the insights array as metric strings."""
    metrics = []
    for item in insights:
        if isinstance(item, dict) and item.get("type") == "text":
            content = item.get("content", "")
            if content:
                metrics.append(str(content))
    return metrics[:5]


def _extract_quality_notes(schema: dict) -> list[str]:
    """Generate data quality notes from schema metadata."""
    notes = []
    for col in schema.get("columns", []):
        name = col.get("name", "")
        # Note high-null columns from the schema analysis
        # (actual null_pct lives in metadata, not schema AI response)
        role = col.get("business_role", "")
        if role == "unknown":
            notes.append(f"Column '{name}' has an unclear business meaning — consider renaming.")

    if not notes:
        notes.append("No major data quality issues detected.")
    return notes
