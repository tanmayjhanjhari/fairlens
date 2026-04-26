"""
FairLens — Upload Router

POST /api/upload       — ingest a CSV dataset
POST /api/upload-model — ingest a serialised scikit-learn model
"""

from __future__ import annotations

import io
import uuid
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, status

router = APIRouter(tags=["Upload"])

# ── Allowed MIME / extensions ─────────────────────────────────────────────────
CSV_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
}
MODEL_EXTENSIONS = {".pkl", ".joblib"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Accept a CSV file and store it in the in-memory session store.

    Returns column metadata, a 5-row preview, and dtype classification.
    """
    # ── Read file bytes ───────────────────────────────────────────────────────
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV parsed but contains no rows.",
        )

    # ── Build metadata ────────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    row_count, col_count = df.shape

    # dtype strings for each column
    dtypes: dict[str, str] = {col: str(df[col].dtype) for col in df.columns}

    # Classify columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # 5-row preview as list of dicts (NaN → None for JSON safety)
    preview = df.head(5).where(pd.notna(df.head(5)), other=None).to_dict(
        orient="records"
    )

    # ── Persist in session store ──────────────────────────────────────────────
    request.app.state.sessions[session_id] = {
        "df": df,
        "filename": file.filename or "upload.csv",
        "row_count": row_count,
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "row_count": row_count,
        "col_count": col_count,
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "preview": preview,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload-model
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload-model", status_code=status.HTTP_201_CREATED)
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Accept a .pkl or .joblib model file.

    The model must be a scikit-learn-compatible estimator with a ``predict``
    method.  Optionally associates the model with an existing ``session_id``.
    """
    # ── Extension check ───────────────────────────────────────────────────────
    filename: str = file.filename or ""
    if not any(filename.endswith(ext) for ext in MODEL_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{filename}'. "
                "Only .pkl and .joblib files are accepted."
            ),
        )

    # ── Read & deserialise ────────────────────────────────────────────────────
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded model file is empty.",
        )

    try:
        model = joblib.load(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not deserialise model: {exc}",
        )

    # ── Validate that the model has a predict method ──────────────────────────
    if not callable(getattr(model, "predict", None)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Loaded object does not have a `predict` method. "
                "Only scikit-learn-compatible models are supported."
            ),
        )

    # ── Introspect model ──────────────────────────────────────────────────────
    model_type = type(model).__name__
    feature_names: list[str] | None = None
    feature_names_available = False

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        feature_names_available = True

    n_features: int | None = None
    if feature_names is not None:
        n_features = len(feature_names)
    elif hasattr(model, "n_features_in_"):
        n_features = int(model.n_features_in_)

    # ── Generate IDs and persist ──────────────────────────────────────────────
    model_id = str(uuid.uuid4())

    # Store model in the session if session_id was provided; else top-level
    sessions: dict = request.app.state.sessions
    if session_id and session_id in sessions:
        sessions[session_id]["model"] = model
        sessions[session_id]["model_id"] = model_id
    else:
        # Store as a standalone entry keyed by model_id
        sessions[model_id] = {
            "model": model,
            "model_id": model_id,
            "filename": filename,
        }

    return {
        "model_id": model_id,
        "model_type": model_type,
        "n_features": n_features,
        "feature_names": feature_names,
        "feature_names_available": feature_names_available,
    }
