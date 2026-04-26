"""
FairLens — Mitigate Router

POST /api/mitigate
  Runs Reweighing and Threshold Adjustment simultaneously,
  returns side-by-side before/after fairness and performance metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

from services.bias_engine import BiasEngine

router = APIRouter(tags=["Mitigation"])
engine = BiasEngine()


class MitigateRequest(BaseModel):
    session_id: str
    techniques: list[str] = Field(default=["reweighing", "threshold"])


@router.post("/mitigate", status_code=status.HTTP_200_OK)
async def mitigate(
    body: MitigateRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Run one or both mitigation techniques on the session dataset.

    Technique options: "reweighing", "threshold"
    """
    sessions: dict = request.app.state.sessions

    if body.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found.",
        )

    session = sessions[body.session_id]

    if "bias_results" not in session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Run /api/analyze before requesting mitigation.",
        )

    df: pd.DataFrame = session["df_with_predictions"].copy()
    target_col: str = session["target_col"]
    sensitive_attrs: list[str] = session["sensitive_attrs"]
    original_metrics = session["bias_results"]

    # Build feature set (numeric, exclude target and predictions)
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in (target_col, "__predictions__")
    ]

    if not feature_cols:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No numeric feature columns available for mitigation modelling.",
        )

    df_clean = df.dropna(subset=[target_col] + sensitive_attrs + feature_cols)
    X = df_clean[feature_cols].values
    y = df_clean[target_col].astype(int).values

    # Use the first sensitive attribute as the fairness constraint attribute
    primary_attr = sensitive_attrs[0]
    sensitive_series = df_clean[primary_attr]

    results: dict[str, Any] = {
        "session_id": body.session_id,
        "original": _extract_original(original_metrics, y, y),
    }

    # ── Reweighing via ExponentiatedGradient ─────────────────────────────────
    if "reweighing" in body.techniques:
        try:
            base_clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            mitigator = ExponentiatedGradient(
                base_clf,
                constraints=DemographicParity(),
                eps=0.01,
            )
            mitigator.fit(X, y, sensitive_features=sensitive_series)
            y_pred_rew = mitigator.predict(X)

            rew_bias = engine.analyze(
                df=df_clean.assign(__predictions__=y_pred_rew),
                target_col=target_col,
                sensitive_attrs=sensitive_attrs,
                use_predictions=True,
            )
            results["reweighing"] = {
                "metrics": _serialise(rew_bias),
                "performance": _perf(y, y_pred_rew),
            }
        except Exception as exc:
            results["reweighing"] = {"error": str(exc)}

    # ── Threshold Adjustment via ThresholdOptimizer ───────────────────────────
    if "threshold" in body.techniques:
        try:
            base_clf2 = LogisticRegression(max_iter=1000, solver="lbfgs")
            base_clf2.fit(X, y)
            postprocess = ThresholdOptimizer(
                estimator=base_clf2,
                constraints="demographic_parity",
                predict_method="predict_proba",
                objective="balanced_accuracy_score",
            )
            postprocess.fit(X, y, sensitive_features=sensitive_series)
            y_pred_thr = postprocess.predict(X, sensitive_features=sensitive_series)

            thr_bias = engine.analyze(
                df=df_clean.assign(__predictions__=y_pred_thr),
                target_col=target_col,
                sensitive_attrs=sensitive_attrs,
                use_predictions=True,
            )
            results["threshold"] = {
                "metrics": _serialise(thr_bias),
                "performance": _perf(y, y_pred_thr),
            }
        except Exception as exc:
            results["threshold"] = {"error": str(exc)}

    # ── Recommend best technique ──────────────────────────────────────────────
    results["recommendation"] = _recommend(results)

    session["mitigation_results"] = results
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _perf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _extract_original(
    bias_results: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    return {
        "audit_score": bias_results.get("audit_score"),
        "grade": bias_results.get("grade"),
        "overall_severity": bias_results.get("overall_severity"),
        "performance": _perf(y_true, y_pred),
    }


def _recommend(results: dict) -> str:
    candidates = {}
    for technique in ("reweighing", "threshold"):
        if technique in results and "metrics" in results[technique]:
            score = results[technique]["metrics"].get("audit_score", 0) or 0
            candidates[technique] = score
    if not candidates:
        return "none"
    return max(candidates, key=lambda k: candidates[k])


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
