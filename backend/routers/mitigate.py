"""
FairLens — Mitigate Router

POST /api/mitigate
  Runs BiasMitigator (reweighing + threshold adjustment).
  If validation flagged fallback_needed, also runs FairlearnFallback
  to enrich the response with extended fairness metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from services.mitigator import BiasMitigator
from services.fairlearn_fallback import FairlearnFallback

router = APIRouter(tags=["Mitigation"])

mitigator = BiasMitigator()
fallback = FairlearnFallback()


class MitigateRequest(BaseModel):
    session_id: str
    target_col: str
    sensitive_attr: str


@router.post("/mitigate", status_code=status.HTTP_200_OK)
async def mitigate(
    body: MitigateRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Run reweighing and threshold-adjustment mitigation strategies.

    If the dataset requires a Fairlearn fallback (continuous or multiclass
    target), fairlearn metrics are also computed and included in the response.
    """
    sessions: dict = request.app.state.sessions

    # ── Session lookup ────────────────────────────────────────────────────────
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found. Upload a CSV first.",
        )

    session = sessions[body.session_id]
    df: pd.DataFrame = session["df"].copy()

    # ── Column validation ─────────────────────────────────────────────────────
    for col, label in [(body.target_col, "target"), (body.sensitive_attr, "sensitive attribute")]:
        if col not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"The {label} column '{col}' was not found in the dataset.",
            )

    # ── Check if fairlearn fallback is needed ─────────────────────────────────
    validation: dict = session.get("validation", {})
    fallback_needed: bool = validation.get("fallback_needed", False)

    fairlearn_result: dict | None = None
    if fallback_needed:
        try:
            fairlearn_result = fallback.analyze(
                df=df,
                target_col=body.target_col,
                sensitive_attr=body.sensitive_attr,
            )
        except Exception as exc:
            fairlearn_result = {"error": str(exc), "fallback_used": True}

    # ── Run BiasMitigator ─────────────────────────────────────────────────────
    # BiasMitigator works on any tabular data regardless of fallback flag
    try:
        mitigation_results = mitigator.run_both(
            df=df,
            target_col=body.target_col,
            sensitive_attr=body.sensitive_attr,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mitigation failed: {exc}",
        )

    # ── Persist in session ────────────────────────────────────────────────────
    session["mitigation_results"] = mitigation_results
    session["fairlearn_results"] = fairlearn_result

    # ── Build response ────────────────────────────────────────────────────────
    response: dict[str, Any] = {
        "session_id": body.session_id,
        "reweigh": _serialise(mitigation_results["reweigh"]),
        "threshold": _serialise(mitigation_results["threshold"]),
        "winner": mitigation_results["winner"],
        "fairlearn_used": fallback_needed,
    }

    if fairlearn_result is not None:
        response["fairlearn_metrics"] = _serialise(fairlearn_result)

    return response


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _serialise(obj: Any) -> Any:
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
