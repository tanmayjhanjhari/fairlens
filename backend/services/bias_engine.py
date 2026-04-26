"""
FairLens — Bias Analysis Engine

Computes SPD, DI, EOD, AOD per sensitive attribute with bootstrapped
confidence intervals, severity labels, and a composite Audit Score.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

import numpy as np
import pandas as pd

# Suppress noisy sklearn warnings during bootstrap resamples
_warnings.filterwarnings("ignore", category=RuntimeWarning)


class BiasEngine:
    """
    Compute fairness metrics for a DataFrame.

    All metrics follow the convention that *privileged* is the largest
    demographic group by count.  This is pragmatic and avoids requiring the
    caller to know which group is historically advantaged.
    """

    BOOTSTRAP_N: int = 200
    BOOTSTRAP_SEED: int = 42

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attrs: list[str],
        use_predictions: bool = False,
    ) -> dict[str, Any]:
        """
        Run full bias analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset.  If ``use_predictions`` is True the frame must contain a
            column named ``__predictions__`` with model-generated labels.
        target_col : str
            Ground-truth label column.
        sensitive_attrs : list[str]
            Protected-attribute columns to analyse.
        use_predictions : bool
            When True, metrics are computed against ``__predictions__`` instead
            of ``target_col``.

        Returns
        -------
        dict
            ``metrics_per_attr``, ``audit_score``, ``overall_severity``
        """
        label_col = "__predictions__" if use_predictions else target_col

        metrics_per_attr: dict[str, Any] = {}

        for attr in sensitive_attrs:
            if attr not in df.columns:
                metrics_per_attr[attr] = {
                    "error": f"Column '{attr}' not found in dataset."
                }
                continue

            # Drop rows where the attribute or label is null
            sub = df[[attr, target_col, label_col]].dropna(
                subset=[attr, label_col]
            )
            if sub.empty:
                metrics_per_attr[attr] = {
                    "error": "No valid rows after dropping nulls."
                }
                continue

            metrics_per_attr[attr] = self._compute_attr_metrics(
                sub, attr, target_col, label_col
            )

        # ── Audit Score ───────────────────────────────────────────────────────
        spd_values = [
            abs(m["spd"])
            for m in metrics_per_attr.values()
            if isinstance(m, dict) and "spd" in m
        ]
        if spd_values:
            audit_score = max(0.0, 100.0 - (np.mean(spd_values) * 100))
        else:
            audit_score = 100.0

        # Grade
        audit_score_rounded = round(audit_score, 2)
        grade = self._grade(audit_score_rounded)

        # Overall severity
        severities = [
            m["severity"]
            for m in metrics_per_attr.values()
            if isinstance(m, dict) and "severity" in m
        ]
        if "high" in severities:
            overall_severity = "high"
        elif "medium" in severities:
            overall_severity = "medium"
        else:
            overall_severity = "low"

        return {
            "metrics_per_attr": metrics_per_attr,
            "audit_score": audit_score_rounded,
            "grade": grade,
            "overall_severity": overall_severity,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_attr_metrics(
        self,
        sub: pd.DataFrame,
        attr: str,
        target_col: str,
        label_col: str,
    ) -> dict[str, Any]:
        """Compute all metrics for a single sensitive attribute."""

        # ── Group statistics ──────────────────────────────────────────────────
        group_stats = self._group_stats(sub, attr, label_col)

        # Privileged = largest group
        privileged = max(group_stats, key=lambda g: group_stats[g]["count"])
        unprivileged_groups = [g for g in group_stats if g != privileged]

        if not unprivileged_groups:
            # Only one group — metrics are meaningless
            return {
                "error": f"'{attr}' has only one unique group after null removal.",
                "group_stats": group_stats,
            }

        # Pick the *most disadvantaged* unprivileged group for scalar metrics
        unprivileged = min(
            unprivileged_groups,
            key=lambda g: group_stats[g]["positive_rate"],
        )

        pr_priv = group_stats[privileged]["positive_rate"]
        pr_unpriv = group_stats[unprivileged]["positive_rate"]

        # ── SPD ───────────────────────────────────────────────────────────────
        spd = float(pr_priv - pr_unpriv)

        # ── DI ────────────────────────────────────────────────────────────────
        if pr_priv == 0:
            di = float("nan")
        else:
            di = float(pr_unpriv / pr_priv)

        # ── EOD & AOD ─────────────────────────────────────────────────────────
        eod, aod = self._equal_opportunity(sub, attr, target_col, label_col, privileged, unprivileged)

        # ── Severity ──────────────────────────────────────────────────────────
        severity = self._severity(spd)

        # ── Legal flag ────────────────────────────────────────────────────────
        legal_flag = (not np.isnan(di)) and (di < 0.8)

        # ── Bootstrapped CI on SPD ────────────────────────────────────────────
        ci, significant = self._bootstrap_spd_ci(sub, attr, label_col, privileged, unprivileged)

        # ── Assemble ──────────────────────────────────────────────────────────
        result: dict[str, Any] = {
            "privileged_group": str(privileged),
            "unprivileged_group": str(unprivileged),
            "group_stats": {
                str(k): v for k, v in group_stats.items()
            },
            "spd": round(spd, 4),
            "di": round(di, 4) if not np.isnan(di) else None,
            "eod": round(eod, 4) if eod is not None else None,
            "aod": round(aod, 4) if aod is not None else None,
            "severity": severity,
            "legal_flag": legal_flag,
            "bootstrapped_ci": ci,
            "statistically_significant": significant,
        }

        return result

    # ── Group stats ───────────────────────────────────────────────────────────

    def _group_stats(
        self,
        sub: pd.DataFrame,
        attr: str,
        label_col: str,
    ) -> dict[Any, dict[str, float]]:
        total = len(sub)
        stats: dict[Any, dict[str, float]] = {}
        for group_val, grp in sub.groupby(attr):
            count = len(grp)
            positive_rate = float(grp[label_col].mean())
            stats[group_val] = {
                "count": count,
                "positive_rate": round(positive_rate, 4),
                "pct_of_total": round(count / total * 100, 2),
            }
        return stats

    # ── EOD / AOD ─────────────────────────────────────────────────────────────

    def _equal_opportunity(
        self,
        sub: pd.DataFrame,
        attr: str,
        target_col: str,
        label_col: str,
        privileged: Any,
        unprivileged: Any,
    ) -> tuple[float | None, float | None]:
        """
        Return (EOD, AOD).

        EOD = TPR_privileged - TPR_unprivileged
        AOD = mean(TPR_diff, FPR_diff)

        If ground-truth labels equal the label column (label_col == target_col
        or no separate predictions exist), EOD/AOD are computed against the
        single label column treating it as both prediction and truth —
        in that case TPR=1 for all groups, so we fall back to None to avoid
        misleading 0.0 values.
        """
        if label_col == target_col:
            # No separate model predictions — cannot compute TPR/FPR meaningfully
            return None, None

        def tpr_fpr(mask: pd.Series) -> tuple[float, float]:
            grp = sub[mask]
            actual = grp[target_col]
            pred = grp[label_col]
            tp = int(((pred == 1) & (actual == 1)).sum())
            fn = int(((pred == 0) & (actual == 1)).sum())
            fp = int(((pred == 1) & (actual == 0)).sum())
            tn = int(((pred == 0) & (actual == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return tpr, fpr

        priv_mask = sub[attr] == privileged
        unpriv_mask = sub[attr] == unprivileged

        tpr_priv, fpr_priv = tpr_fpr(priv_mask)
        tpr_unpriv, fpr_unpriv = tpr_fpr(unpriv_mask)

        eod = tpr_priv - tpr_unpriv
        aod = ((tpr_priv - tpr_unpriv) + (fpr_priv - fpr_unpriv)) / 2.0

        return float(eod), float(aod)

    # ── Bootstrapped CI ───────────────────────────────────────────────────────

    def _bootstrap_spd_ci(
        self,
        sub: pd.DataFrame,
        attr: str,
        label_col: str,
        privileged: Any,
        unprivileged: Any,
    ) -> tuple[dict[str, float], bool]:
        """
        Compute 95% bootstrapped confidence interval for SPD.

        Returns (ci_dict, statistically_significant).
        """
        rng = np.random.default_rng(self.BOOTSTRAP_SEED)
        n = len(sub)
        spd_samples: list[float] = []

        for _ in range(self.BOOTSTRAP_N):
            sample = sub.iloc[rng.integers(0, n, size=n)]
            priv_rate = float(sample.loc[sample[attr] == privileged, label_col].mean())
            unpriv_rate = float(sample.loc[sample[attr] == unprivileged, label_col].mean())
            if np.isnan(priv_rate) or np.isnan(unpriv_rate):
                continue
            spd_samples.append(priv_rate - unpriv_rate)

        if len(spd_samples) < 10:
            return {"low_95": None, "high_95": None}, False

        low_95 = float(np.percentile(spd_samples, 2.5))
        high_95 = float(np.percentile(spd_samples, 97.5))
        # Statistically significant if CI does NOT cross zero
        significant = not (low_95 <= 0 <= high_95)

        return (
            {"low_95": round(low_95, 4), "high_95": round(high_95, 4)},
            significant,
        )

    # ── Severity & Grade ──────────────────────────────────────────────────────

    @staticmethod
    def _severity(spd: float) -> str:
        abs_spd = abs(spd)
        if abs_spd < 0.1:
            return "low"
        if abs_spd < 0.2:
            return "medium"
        return "high"

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 80:
            return "A"
        if score >= 60:
            return "B"
        if score >= 40:
            return "C"
        return "F"
