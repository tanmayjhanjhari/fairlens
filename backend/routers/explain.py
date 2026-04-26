"""
FairLens — Explain Router

POST /api/explain
  Runs BiasExplainer for a single sensitive attribute and stores the
  explanation in the session for use by the PDF report and Copilot.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from services.explainer import BiasExplainer

router = APIRouter(tags=["Explanation"])

explainer = BiasExplainer()


class ExplainRequest(BaseModel):
    session_id: str
    target_col: str
    sensitive_attr: str


@router.post("/explain", status_code=status.HTTP_200_OK)
async def explain(
    body: ExplainRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Run bias explanation analysis for a single sensitive attribute.

    Returns correlation, proxy features, data imbalance, historical skew,
    positive rate gap, and a plain-English reason string.
    """
    sessions: dict = request.app.state.sessions

    # ── Session lookup ────────────────────────────────────────────────────────
    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found. Upload a CSV first.",
        )

    session = sessions[body.session_id]

    import pandas as pd
    df: pd.DataFrame = session["df"]

    # ── Column validation ─────────────────────────────────────────────────────
    if body.target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{body.target_col}' not found in dataset.",
        )

    if body.sensitive_attr not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Sensitive attribute '{body.sensitive_attr}' not found in dataset.",
        )

    # ── Run explainer ─────────────────────────────────────────────────────────
    try:
        explanation = explainer.explain(
            df=df,
            target_col=body.target_col,
            sensitive_attr=body.sensitive_attr,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {exc}",
        )

    # ── Persist in session (keyed by attribute for multi-attr support) ────────
    if "explanations" not in session:
        session["explanations"] = {}
    session["explanations"][body.sensitive_attr] = explanation

    return {"session_id": body.session_id, **explanation}
