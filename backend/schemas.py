"""
models/schemas.py
All Pydantic request/response models for the DataMind AI API.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional


# ─────────────────────────────────────────────
#  CSV / FILE MODELS
# ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    file_id:           str
    dataset_id:        Optional[str] = None
    dataset_name:      Optional[str] = None
    filename:          str
    original_filename: str
    size_kb:           float
    columns:           list[str]
    row_count:         int
    message:           str = "File uploaded successfully"


class FileListItem(BaseModel):
    file_id:           str
    filename:          str
    original_filename: str
    size_kb:           float
    row_count:         int = 0
    uploaded_at:       Optional[Any] = None


class ReadCSVResponse(BaseModel):
    file_id:    str
    total_rows: int
    offset:     int
    limit:      int
    columns:    list[str]
    data:       list[dict[str, Any]]
    returned:   int


class ColumnMeta(BaseModel):
    name:          str
    dtype:         str
    null_count:    int
    null_pct:      float
    unique_count:  int
    sample_values: list[Any]
    stats:         Optional[dict[str, float]] = None
    likely_date:   Optional[bool]             = None
    ai_description: Optional[str]             = None   # filled by schema service


class MetadataResponse(BaseModel):
    file_id:        str
    file_hash_md5:  str
    shape:          dict[str, int]
    column_count:   int
    row_count:      int
    duplicate_rows: int
    memory_usage_kb: float
    columns:        list[ColumnMeta]


class DeleteResponse(BaseModel):
    deleted: bool
    file_id: str
    message: str = "File deleted successfully"


class DatasetRecord(BaseModel):
    dataset_id:        str
    file_id:           str
    dataset_name:      str
    original_filename: str
    stored_filename:   str
    upload_date:       str
    updated_at:        str
    owner:             str
    row_count:         int
    column_count:      int
    file_size:         int
    file_type:         str


class DatasetRenameRequest(BaseModel):
    dataset_name: str = Field(..., min_length=1, max_length=120)


class DatasetDeleteResponse(BaseModel):
    deleted: bool
    dataset_id: str
    message: str = "Dataset deleted successfully"


class DatasetEvent(BaseModel):
    event_id:   str
    dataset_id: str
    action:     str
    owner:      str
    timestamp:  str
    details:    dict[str, Any] = Field(default_factory=dict)


class DatasetMetadataResponse(BaseModel):
    dataset: DatasetRecord
    metadata: MetadataResponse


# ─────────────────────────────────────────────
#  SCHEMA INTELLIGENCE MODELS
# ─────────────────────────────────────────────

class ColumnDescription(BaseModel):
    name:            str
    ai_description:  str
    business_role:   str    # e.g. "metric", "dimension", "identifier", "date"
    suggested_label: str    # human-friendly label


class SchemaAnalysis(BaseModel):
    file_id:          str
    dataset_summary:  str               # one-paragraph AI description of the dataset
    likely_domain:    str               # e.g. "E-commerce sales", "HR data"
    suggested_questions: list[str]      # 5 questions users could ask
    columns:          list[ColumnDescription]


# ─────────────────────────────────────────────
#  AI QUERY MODELS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    file_id: str   = Field(..., description="UUID of the uploaded dataset")
    question: str  = Field(..., description="Natural language question about the data",
                           min_length=3, max_length=1000)
    max_rows: Optional[int] = Field(500, description="Max data rows to pass to AI")


class InsightItem(BaseModel):
    type:    str          # "text" | "table" | "chart_suggestion"
    content: Any


class QueryResponse(BaseModel):
    file_id:       str
    question:      str
    answer:        str                    # Main AI narrative answer
    insights:      list[InsightItem]      # Structured breakdown
    suggested_followups: list[str]        # Follow-up questions
    rows_analyzed: int
    model_used:    str
    business_summary: Optional[str] = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    execution_time_ms: Optional[int] = None
    analysis_plan: dict[str, Any] = Field(default_factory=dict)
    generated_code: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str
    message: str
    version: str = "1.0.0"
