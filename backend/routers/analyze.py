"""
ByUs — Analyze Router

POST /api/analyze
  1. Validate the dataset with DataValidator
  2. Optionally run model.predict() for prediction-based bias
  3. Run BiasEngine.analyze()
  4. Persist and return results
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from services.validator import DataValidator
from services.bias_engine import BiasEngine

router = APIRouter(tags=["Analysis"])

validator = DataValidator()
engine = BiasEngine()


# ── Request body ──────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    session_id: str
    target_col: str
    sensitive_attrs: list[str] = Field(min_length=1)
    model_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/analyze
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze(
    body: AnalyzeRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Run the full ByUs bias analysis pipeline.

    Steps:
    1. Retrieve the dataset from the session store.
    2. Validate target column and sensitive attributes with DataValidator.
    3. If model_id is supplied, run model.predict() and attach predictions
       as the ``__predictions__`` column.
    4. Run BiasEngine.analyze() and store results back in the session.
    """
    sessions: dict = request.app.state.sessions

    # ── 1. Retrieve session data ───────────────────────────────────────────────
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found. Please upload a CSV first.",
        )

    session = sessions[body.session_id]
    df: pd.DataFrame = session["df"].copy()

    # ── Basic column existence checks ─────────────────────────────────────────
    if body.target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{body.target_col}' not found in dataset.",
        )

    missing_attrs = [a for a in body.sensitive_attrs if a not in df.columns]
    if missing_attrs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Sensitive attribute(s) not found in dataset: {missing_attrs}",
        )

    # ── 2. Validate ───────────────────────────────────────────────────────────
    validation = validator.validate(df, body.target_col, body.sensitive_attrs)

    # Hard-stop if the dataset has fewer than the absolute minimum rows
    if validation["row_count"] < DataValidator.MIN_ROWS_ABSOLUTE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Dataset has only {validation['row_count']} rows. "
                f"A minimum of {DataValidator.MIN_ROWS_ABSOLUTE} rows is required."
            ),
        )

    # ── 3. Optional model predictions ─────────────────────────────────────────
    model_used = False
    use_predictions = False

    if body.model_id:
        model_session = _find_model(sessions, body.model_id, body.session_id)
        if model_session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{body.model_id}' not found. Please upload the model first.",
            )

        model = model_session["model"]
        feature_names: list[str] | None = getattr(model, "feature_names_in_", None)

        if feature_names is not None:
            feature_cols = list(feature_names)
        else:
            # Fallback: use all numeric columns excluding the target
            feature_cols = [
                c for c in df.select_dtypes(include="number").columns
                if c != body.target_col
            ]

        # Ensure all feature columns exist
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Model expects features {missing_features} "
                    "that are not present in the uploaded dataset."
                ),
            )

        try:
            X = df[feature_cols].copy()
            
            # Auto-encode any string/categorical columns to prevent "could not convert string to float" errors
            from sklearn.preprocessing import LabelEncoder
            for c in X.columns:
                if X[c].dtype == 'object' or X[c].dtype.name == 'category' or X[c].dtype == 'bool':
                    try:
                        # Try to cast to float first in case it's just numeric strings
                        X[c] = X[c].astype(float)
                    except ValueError:
                        # Fallback to label encoding (alphabetical)
                        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

            X = X.fillna(0)
            predictions = model.predict(X)
            df["__predictions__"] = predictions
            use_predictions = True
            model_used = True
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model prediction failed: {exc}",
            )
    else:
        # Without a model, treat the target column as the "prediction"
        df["__predictions__"] = df[body.target_col]

    # ── 4. Run bias engine ────────────────────────────────────────────────────
    try:
        bias_results = engine.analyze(
            df=df,
            target_col=body.target_col,
            sensitive_attrs=body.sensitive_attrs,
            use_predictions=use_predictions,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bias analysis failed: {exc}",
        )

    # ── 5. Persist results ────────────────────────────────────────────────────
    session["validation"] = validation
    session["bias_results"] = bias_results
    session["target_col"] = body.target_col
    session["sensitive_attrs"] = body.sensitive_attrs
    session["df_with_predictions"] = df  # needed for mitigation

    # ── 6. Build response ─────────────────────────────────────────────────────
    response: dict[str, Any] = {
        "session_id": body.session_id,
        "validation": validation,
        "metrics_per_attr": _serialise(bias_results["metrics_per_attr"]),
        "audit_score": bias_results["audit_score"],
        "grade": bias_results["grade"],
        "overall_severity": bias_results["overall_severity"],
        "model_used": model_used,
    }

    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_model(
    sessions: dict,
    model_id: str,
    session_id: str,
) -> dict | None:
    """
    Look for the model in the given session first, then in the top-level store
    (model may have been uploaded independently with only a model_id key).
    """
    # Check inside the session
    session = sessions.get(session_id, {})
    if session.get("model_id") == model_id and "model" in session:
        return session

    # Check standalone model entry
    standalone = sessions.get(model_id)
    if standalone and "model" in standalone:
        return standalone

    return None


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python for JSON safety."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _serialise(obj.tolist())
    return obj
