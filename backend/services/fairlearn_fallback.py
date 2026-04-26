"""
FairLens — Fairlearn Fallback Service

Handles datasets that are not binary classification:
  - Continuous targets: binarized at the median
  - Multiclass targets: one-vs-rest per class

Uses fairlearn.metrics.MetricFrame for group-level analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, selection_rate
from sklearn.preprocessing import LabelEncoder


class FairlearnFallback:
    """
    Fairlearn-powered analysis for continuous and multiclass targets.
    Always sets fallback_used=True and engine="fairlearn" in the result.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> dict[str, Any]:
        """
        Run fairlearn-based group analysis.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
        sensitive_attr : str

        Returns
        -------
        dict with keys:
            metrics_per_class, warnings, fallback_used, engine,
            demographic_parity_difference, equalized_odds_difference
        """
        warnings: list[str] = []
        work = df.copy().dropna(subset=[target_col, sensitive_attr])

        # ── Encode sensitive attribute if categorical ──────────────────────────
        sensitive_series = work[sensitive_attr]

        # ── Determine target type ─────────────────────────────────────────────
        n_unique = work[target_col].nunique()
        is_continuous = (
            pd.api.types.is_numeric_dtype(work[target_col]) and n_unique > 10
        )
        is_multiclass = (not is_continuous) and (n_unique > 2)

        # ── Binarize continuous target at median ──────────────────────────────
        if is_continuous:
            median_val = work[target_col].median()
            work["__binary_target__"] = (work[target_col] > median_val).astype(int)
            analysis_target = "__binary_target__"
            warnings.append(
                f"Target '{target_col}' is continuous. It has been binarized at the "
                f"median value ({median_val:.4f}) for fairness analysis."
            )
        else:
            analysis_target = target_col

        y = work[analysis_target].values
        sensitive = sensitive_series.values

        results_per_class: dict[str, Any] = {}

        if is_multiclass:
            # ── One-vs-rest per class ─────────────────────────────────────────
            classes = sorted(work[target_col].unique())
            for cls in classes:
                y_binary = (work[target_col] == cls).astype(int).values
                cls_result = self._run_metricframe(
                    y_true=y_binary,
                    y_pred=y_binary,  # use ground-truth labels directly (no model)
                    sensitive=sensitive,
                )
                results_per_class[str(cls)] = cls_result
        else:
            # ── Binary (or binarized continuous) ──────────────────────────────
            results_per_class["binary"] = self._run_metricframe(
                y_true=y,
                y_pred=y,
                sensitive=sensitive,
            )

        # ── Top-level fairlearn metrics ───────────────────────────────────────
        try:
            dpd = float(
                demographic_parity_difference(
                    y_true=y,
                    y_pred=y,
                    sensitive_features=sensitive,
                )
            )
        except Exception:
            dpd = None

        try:
            eod = float(
                equalized_odds_difference(
                    y_true=y,
                    y_pred=y,
                    sensitive_features=sensitive,
                )
            )
        except Exception:
            eod = None

        return {
            "metrics_per_class": results_per_class,
            "demographic_parity_difference": round(dpd, 4) if dpd is not None else None,
            "equalized_odds_difference": round(eod, 4) if eod is not None else None,
            "warnings": warnings,
            "fallback_used": True,
            "engine": "fairlearn",
            "is_continuous_binarized": is_continuous,
            "is_multiclass": is_multiclass,
        }

    # ── MetricFrame helper ────────────────────────────────────────────────────

    @staticmethod
    def _run_metricframe(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> dict[str, Any]:
        """
        Build a Fairlearn MetricFrame and extract per-group metrics.
        """
        try:
            mf = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "selection_rate": selection_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )

            per_group = mf.by_group.reset_index()
            per_group_dict = per_group.to_dict(orient="records")

            overall = mf.overall.to_dict()
            difference = mf.difference().to_dict()

            return {
                "overall": {k: round(float(v), 4) for k, v in overall.items()},
                "per_group": [
                    {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else str(v))
                     for k, v in row.items()}
                    for row in per_group_dict
                ],
                "difference": {k: round(float(v), 4) for k, v in difference.items()},
            }
        except Exception as exc:
            return {"error": str(exc)}
