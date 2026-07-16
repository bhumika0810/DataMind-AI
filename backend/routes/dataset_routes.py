"""
routes/dataset_routes.py
Dataset management endpoints.
"""

import logging

from fastapi import APIRouter, Header, HTTPException, Query

from schemas import (
    DatasetDeleteResponse,
    DatasetEvent,
    DatasetMetadataResponse,
    DatasetRecord,
    DatasetRenameRequest,
)
from services import csv_services, dataset_services

logger = logging.getLogger("datamind.dataset_routes")
router = APIRouter()


@router.get(
    "",
    response_model=list[DatasetRecord],
    summary="List registered datasets",
)
def list_datasets(
    search: str | None = Query(None, description="Search by dataset name, filename, or file type"),
    owner: str | None = Query(None, description="Filter by owner"),
):
    return dataset_services.list_datasets(search=search, owner=owner)


@router.get(
    "/history",
    response_model=list[DatasetEvent],
    summary="List dataset upload history",
)
def upload_history(limit: int = Query(100, ge=1, le=500)):
    return dataset_services.upload_history(limit=limit)


@router.get(
    "/{dataset_id}",
    response_model=DatasetRecord,
    summary="Get dataset details",
)
def get_dataset(dataset_id: str):
    try:
        return dataset_services.get_dataset(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch(
    "/{dataset_id}",
    response_model=DatasetRecord,
    summary="Rename a dataset",
)
def rename_dataset(
    dataset_id: str,
    request: DatasetRenameRequest,
    owner: str = Header("local_user", alias="X-DataMind-Owner"),
):
    try:
        return dataset_services.rename_dataset(dataset_id, request.dataset_name, owner=owner)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete(
    "/{dataset_id}",
    response_model=DatasetDeleteResponse,
    summary="Delete a registered dataset and its stored file",
)
def delete_dataset(
    dataset_id: str,
    owner: str = Header("local_user", alias="X-DataMind-Owner"),
):
    try:
        csv_services.delete_file(dataset_id)
        dataset_services.unregister_dataset(dataset_id, owner=owner)
        return DatasetDeleteResponse(deleted=True, dataset_id=dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{dataset_id}/metadata",
    response_model=DatasetMetadataResponse,
    summary="Get registered dataset details with extracted metadata",
)
def get_dataset_metadata(dataset_id: str):
    try:
        dataset = dataset_services.get_dataset(dataset_id)
        metadata = csv_services.extract_metadata(dataset_id)
        return DatasetMetadataResponse(dataset=dataset, metadata=metadata)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Dataset metadata failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset metadata failed: {e}")


@router.get(
    "/{dataset_id}/events",
    response_model=list[DatasetEvent],
    summary="Get dataset audit-style event history",
)
def get_dataset_events(dataset_id: str):
    try:
        return dataset_services.dataset_events(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
