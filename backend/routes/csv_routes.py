"""
routes/csv_routes.py

CSV & File Management Endpoints:

  POST   /api/csv/upload          — Upload CSV or Excel file
  GET    /api/csv/files           — List all uploaded files
  GET    /api/csv/{file_id}/read  — Read rows (paginated)
  GET    /api/csv/{file_id}/meta  — Deep metadata extraction
  GET    /api/csv/{file_id}/schema — AI schema intelligence
  DELETE /api/csv/{file_id}       — Delete a file
"""

import logging
from fastapi import APIRouter, Header, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from services import csv_services, dataset_services, schema_services
from schemas import (
    UploadResponse,
    FileListItem,
    ReadCSVResponse,
    MetadataResponse,
    SchemaAnalysis,
    DeleteResponse,
)
from config import settings

logger = logging.getLogger("datamind.csv_routes")
router = APIRouter()

# Max upload size check (FastAPI doesn't enforce this by default)
MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


# ─────────────────────────────────────────────
#  POST /upload
# ─────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a CSV or Excel file",
    description="Accepts .csv, .xlsx, .xls. Stores it server-side and returns a file_id for all future operations.",
)
async def upload_file(
    file: UploadFile = File(...),
    owner: str = Header("local_user", alias="X-DataMind-Owner"),
):
    # Validate content type / extension
    allowed_types = {
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    if file.content_type not in allowed_types and not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload CSV or Excel.",
        )

    file_bytes = await file.read()

    # Size guard
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)//1024} KB). Maximum is {settings.MAX_FILE_SIZE_MB} MB.",
        )

    try:
        result = await csv_services.save_csv(file_bytes, file.filename)
        dataset = dataset_services.register_dataset(result, owner=owner)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return UploadResponse(
        file_id=result["file_id"],
        dataset_id=dataset["dataset_id"],
        dataset_name=dataset["dataset_name"],
        filename=result["filename"],
        original_filename=result["original_filename"],
        size_kb=result["size_kb"],
        columns=result["columns"],
        row_count=result["row_count"],
    )


# ─────────────────────────────────────────────
#  GET /files
# ─────────────────────────────────────────────

@router.get(
    "/files",
    response_model=list[FileListItem],
    summary="List all uploaded files",
)
def list_files():
    datasets = dataset_services.list_datasets()
    if datasets:
        return [
            {
                "file_id": item["file_id"],
                "filename": item["stored_filename"],
                "original_filename": item["original_filename"],
                "size_kb": round(item["file_size"] / 1024, 2),
                "row_count": item["row_count"],
                "uploaded_at": item["upload_date"],
            }
            for item in datasets
        ]
    return csv_services.list_files()


# ─────────────────────────────────────────────
#  GET /{file_id}/read
# ─────────────────────────────────────────────

@router.get(
    "/{file_id}/read",
    response_model=ReadCSVResponse,
    summary="Read rows from an uploaded file (paginated)",
)
def read_file(
    file_id: str,
    limit:  int = Query(100, ge=1, le=5000, description="Rows to return"),
    offset: int = Query(0,   ge=0,          description="Starting row index"),
):
    try:
        return csv_services.read_csv(file_id, limit=limit, offset=offset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─────────────────────────────────────────────
#  GET /{file_id}/meta
# ─────────────────────────────────────────────

@router.get(
    "/{file_id}/meta",
    response_model=MetadataResponse,
    summary="Deep metadata extraction (dtypes, nulls, stats, etc.)",
)
def get_metadata(file_id: str):
    try:
        result = csv_services.extract_metadata(file_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {e}")


# ─────────────────────────────────────────────
#  GET /{file_id}/schema
# ─────────────────────────────────────────────

@router.get(
    "/{file_id}/schema",
    response_model=SchemaAnalysis,
    summary="AI-powered schema intelligence (domain, column descriptions, suggested questions)",
)
async def get_schema_analysis(file_id: str):
    try:
        result = await schema_services.analyse_schema(file_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    except Exception as e:
        logger.error(f"Schema analysis failed for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  DELETE /{file_id}
# ─────────────────────────────────────────────

@router.delete(
    "/{file_id}",
    response_model=DeleteResponse,
    summary="Delete an uploaded file",
)
def delete_file(
    file_id: str,
    owner: str = Header("local_user", alias="X-DataMind-Owner"),
):
    try:
        result = csv_services.delete_file(file_id)
        try:
            dataset_services.unregister_dataset(file_id, owner=owner)
        except FileNotFoundError:
            pass
        return DeleteResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
