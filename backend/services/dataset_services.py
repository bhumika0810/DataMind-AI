"""
services/dataset_services.py
Dataset registry for uploaded files.

Phase 1 intentionally keeps persistence behind this service boundary. The
current implementation uses an atomic JSON registry; a database-backed version
can later preserve this module API.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings

_LOCK = threading.Lock()


def register_dataset(upload_result: dict[str, Any], owner: str = "local_user") -> dict[str, Any]:
    dataset_id = upload_result["file_id"]
    original_filename = upload_result["original_filename"]
    now = _utc_now()

    record = {
        "dataset_id": dataset_id,
        "file_id": dataset_id,
        "dataset_name": Path(original_filename).stem or original_filename,
        "original_filename": original_filename,
        "stored_filename": upload_result["filename"],
        "filepath": upload_result["filepath"],
        "upload_date": now,
        "updated_at": now,
        "owner": _clean_owner(owner),
        "row_count": int(upload_result["row_count"]),
        "column_count": len(upload_result.get("columns", [])),
        "file_size": int(upload_result["size_bytes"]),
        "file_type": Path(original_filename).suffix.lower().lstrip(".") or "csv",
    }

    with _LOCK:
        registry = _read_registry()
        registry["datasets"][dataset_id] = record
        registry.setdefault("events", []).append(
            _event("dataset_uploaded", dataset_id, record["owner"], {"dataset_name": record["dataset_name"]})
        )
        _write_registry(registry)

    return record


def list_datasets(search: str | None = None, owner: str | None = None) -> list[dict[str, Any]]:
    registry = _read_registry()
    datasets = list(registry["datasets"].values())

    if owner:
        owner_key = owner.lower()
        datasets = [item for item in datasets if item.get("owner", "").lower() == owner_key]

    if search:
        term = search.lower()
        datasets = [
            item for item in datasets
            if term in item.get("dataset_name", "").lower()
            or term in item.get("original_filename", "").lower()
            or term in item.get("file_type", "").lower()
        ]

    return sorted(datasets, key=lambda item: item.get("upload_date", ""), reverse=True)


def get_dataset(dataset_id: str) -> dict[str, Any]:
    registry = _read_registry()
    try:
        return registry["datasets"][dataset_id]
    except KeyError:
        raise FileNotFoundError(f"No dataset found with dataset_id: {dataset_id}")


def rename_dataset(dataset_id: str, dataset_name: str, owner: str = "local_user") -> dict[str, Any]:
    cleaned_name = _clean_dataset_name(dataset_name)

    with _LOCK:
        registry = _read_registry()
        if dataset_id not in registry["datasets"]:
            raise FileNotFoundError(f"No dataset found with dataset_id: {dataset_id}")

        registry["datasets"][dataset_id]["dataset_name"] = cleaned_name
        registry["datasets"][dataset_id]["updated_at"] = _utc_now()
        registry.setdefault("events", []).append(
            _event("dataset_renamed", dataset_id, _clean_owner(owner), {"dataset_name": cleaned_name})
        )
        _write_registry(registry)

    return registry["datasets"][dataset_id]


def unregister_dataset(dataset_id: str, owner: str = "local_user") -> None:
    with _LOCK:
        registry = _read_registry()
        if dataset_id not in registry["datasets"]:
            raise FileNotFoundError(f"No dataset found with dataset_id: {dataset_id}")

        registry["datasets"].pop(dataset_id)
        registry.setdefault("events", []).append(
            _event("dataset_deleted", dataset_id, _clean_owner(owner), {})
        )
        _write_registry(registry)


def upload_history(limit: int = 100) -> list[dict[str, Any]]:
    registry = _read_registry()
    upload_events = [
        event for event in registry.get("events", [])
        if event.get("action") == "dataset_uploaded"
    ]
    upload_events.sort(key=lambda event: event.get("timestamp", ""), reverse=True)
    return upload_events[:limit]


def dataset_events(dataset_id: str) -> list[dict[str, Any]]:
    get_dataset(dataset_id)
    registry = _read_registry()
    events = [
        event for event in registry.get("events", [])
        if event.get("dataset_id") == dataset_id
    ]
    return sorted(events, key=lambda event: event.get("timestamp", ""), reverse=True)


def _read_registry() -> dict[str, Any]:
    path = Path(settings.DATASET_REGISTRY_PATH)
    if not path.exists():
        return {"datasets": {}, "events": []}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    data.setdefault("datasets", {})
    data.setdefault("events", [])
    return data


def _write_registry(registry: dict[str, Any]) -> None:
    path = Path(settings.DATASET_REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name

    os.replace(temp_name, path)


def _event(action: str, dataset_id: str, owner: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": f"{dataset_id}:{action}:{_utc_now()}",
        "dataset_id": dataset_id,
        "action": action,
        "owner": owner,
        "timestamp": _utc_now(),
        "details": details,
    }


def _clean_dataset_name(dataset_name: str) -> str:
    cleaned = " ".join(dataset_name.strip().split())
    if not cleaned:
        raise ValueError("Dataset name cannot be empty.")
    if len(cleaned) > 120:
        raise ValueError("Dataset name cannot exceed 120 characters.")
    return cleaned


def _clean_owner(owner: str) -> str:
    cleaned = " ".join((owner or "local_user").strip().split())
    return cleaned[:80] or "local_user"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
