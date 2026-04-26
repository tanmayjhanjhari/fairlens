"""
FairLens — Bias Mitigator Service

Runs two mitigation strategies in parallel:
  1. Reweighing  — assigns sample weights to balance group × label frequencies
  2. Threshold Adjustment — per-group optimal decision thresholds via scipy

Returns before/after metrics, effect deltas, and a winner recommendation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class BiasMitigator:
    """Run reweighing and threshold-adjustment mitigation on tabular data."""

    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.30

    # ── Public API ────────────────────────────────────────────────────────────

    def run_both(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> dict[str, Any]:
        """
        Run both mitigation strategies and return a unified comparison.

        Returns
        -------
        dict with keys: reweigh, threshold, winner
        """
        rew = self.reweigh(df, target_col, sensitive_attr)
        thr = self.threshold_adjust(df, target_col, sensitive_attr)

        # Winner: best SPD reduction while keeping accuracy drop < 3%
        winner = self._pick_winner(rew, thr)

        return {
            "reweigh": rew,
            "threshold": thr,
            "winner": winner,
        }

    # ── Reweighing ────────────────────────────────────────────────────────────

    def reweigh(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> dict[str, Any]:
        """
        Compute sample weights based on the ratio of expected to actual
        joint frequency of (group, label), then retrain a classifier.
        """
        work = df.copy().dropna(subset=[target_col, sensitive_attr])

        # ── Weight computation ────────────────────────────────────────────────
        n = len(work)
        p_group = work[sensitive_attr].value_counts(normalize=True)
        p_label = work[target_col].value_counts(normalize=True)

        weights = np.ones(n)
        weights_summary: list[dict] = []

        for group_val, grp_idx in work.groupby(sensitive_attr).groups.items():
            for label_val in work[target_col].unique():
                mask = (work[sensitive_attr] == group_val) & (
                    work[target_col] == label_val
                )
                actual_freq = mask.mean()
                expected_freq = float(p_group.get(group_val, 0)) * float(
                    p_label.get(label_val, 0)
                )
                w = (
                    expected_freq / actual_freq
                    if actual_freq > 0
                    else 1.0
                )
                weights[mask] = w
                weights_summary.append(
                    {
                        "group": str(group_val),
                        "label": str(label_val),
                        "weight": round(w, 4),
                        "actual_freq": round(actual_freq, 4),
                        "expected_freq": round(expected_freq, 4),
                    }
                )

        # ── Before / after metrics ────────────────────────────────────────────
        before = self._compute_metrics(work, target_col, sensitive_attr)
        after = self._compute_metrics(
            work, target_col, sensitive_attr, sample_weight=weights
        )
        eff = self.effects(before, after)

        return {
            "before": before,
            "after": after,
            "effects": eff,
            "weights_summary": weights_summary,
        }

    # ── Threshold Adjustment ──────────────────────────────────────────────────

    def threshold_adjust(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> dict[str, Any]:
        """
        Train a base classifier, then find per-group decision thresholds that
        maximise TPR equality (equal opportunity) across groups using scipy.
        """
        work = df.copy().dropna(subset=[target_col, sensitive_attr])

        X, y, sensitive = self._prepare_features(work, target_col, sensitive_attr)

        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE,
            stratify=y,
        )

        # Train base model
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=self.RANDOM_STATE)
        clf.fit(X_train, y_train)

        # Before metrics (default threshold 0.5)
        y_pred_before = clf.predict(X_test)
        before = self._metrics_from_arrays(y_test, y_pred_before, s_test, work[sensitive_attr])

        # Get probabilities on test set
        proba = clf.predict_proba(X_test)[:, 1]

        # ── Per-group optimal threshold via scipy ─────────────────────────────
        groups = np.unique(s_test)
        thresholds: dict[str, float] = {}

        # Target TPR: average TPR across groups at default threshold
        def _tpr(y_true, y_pred):
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        target_tpr = float(np.mean([
            _tpr(y_test[s_test == g], y_pred_before[s_test == g])
            for g in groups
        ]))

        for group in groups:
            mask = s_test == group
            y_g = y_test[mask]
            p_g = proba[mask]

            def objective(threshold: float) -> float:
                pred = (p_g >= threshold).astype(int)
                tpr = _tpr(y_g, pred)
                return abs(tpr - target_tpr)

            result = minimize_scalar(objective, bounds=(0.01, 0.99), method="bounded")
            thresholds[str(group)] = round(float(result.x), 4)

        # Apply per-group thresholds
        y_pred_after = np.zeros(len(y_test), dtype=int)
        for group, threshold in thresholds.items():
            mask = s_test == group
            y_pred_after[mask] = (proba[mask] >= threshold).astype(int)

        after = self._metrics_from_arrays(y_test, y_pred_after, s_test, work[sensitive_attr])
        eff = self.effects(before, after)

        return {
            "before": before,
            "after": after,
            "effects": eff,
            "thresholds": thresholds,
        }

    # ── Effects ───────────────────────────────────────────────────────────────

    @staticmethod
    def effects(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        """Compute delta metrics between before and after mitigation."""
        def delta(key: str) -> float | None:
            b = before.get(key)
            a = after.get(key)
            if b is None or a is None:
                return None
            return round(float(a) - float(b), 4)

        spd_before = abs(before.get("spd", 0) or 0)
        spd_after = abs(after.get("spd", 0) or 0)
        bias_reduction_pct = (
            round((spd_before - spd_after) / spd_before * 100, 2)
            if spd_before > 0
            else 0.0
        )

        acc_before = before.get("accuracy", 1) or 1
        acc_after = after.get("accuracy", 1) or 1
        accuracy_retained_pct = round(acc_after / acc_before * 100, 2) if acc_before > 0 else 100.0

        return {
            "accuracy_delta": delta("accuracy"),
            "precision_delta": delta("precision"),
            "recall_delta": delta("recall"),
            "f1_delta": delta("f1"),
            "spd_delta": delta("spd"),
            "bias_reduction_pct": bias_reduction_pct,
            "accuracy_retained_pct": accuracy_retained_pct,
        }

    # ── Core metric computation ───────────────────────────────────────────────

    def _compute_metrics(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Stratified 70/30 split → train LogisticRegression (with optional weights) →
        compute fairness + performance metrics on the held-out test set.
        """
        X, y, sensitive = self._prepare_features(df, target_col, sensitive_attr)

        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE,
            stratify=y,
        )

        train_weights = sample_weight[: len(X_train)] if sample_weight is not None else None

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=self.RANDOM_STATE)
        clf.fit(X_train, y_train, sample_weight=train_weights)
        y_pred = clf.predict(X_test)

        return self._metrics_from_arrays(y_test, y_pred, s_test, df[sensitive_attr])

    def _metrics_from_arrays(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        s_test: np.ndarray,
        sensitive_series: pd.Series,
    ) -> dict[str, Any]:
        """Compute all metrics from raw arrays."""
        # Performance
        accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
        precision = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
        recall = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
        f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

        # Positive rates per group
        groups = np.unique(s_test)
        pos_rates: dict[str, float] = {}
        for g in groups:
            mask = s_test == g
            pos_rates[str(g)] = round(float(y_pred[mask].mean()), 4)

        if len(groups) >= 2:
            privileged = max(pos_rates, key=lambda k: pos_rates[k])
            unprivileged = min(pos_rates, key=lambda k: pos_rates[k])
            pr_priv = pos_rates[privileged]
            pr_unpriv = pos_rates[unprivileged]

            spd = round(pr_priv - pr_unpriv, 4)
            di = round(pr_unpriv / pr_priv, 4) if pr_priv > 0 else None

            # TPR per group
            tpr_vals = {}
            fpr_vals = {}
            for g in groups:
                mask = s_test == g
                yg, pg = y_test[mask], y_pred[mask]
                tp = int(((pg == 1) & (yg == 1)).sum())
                fn = int(((pg == 0) & (yg == 1)).sum())
                fp = int(((pg == 1) & (yg == 0)).sum())
                tn = int(((pg == 0) & (yg == 0)).sum())
                tpr_vals[str(g)] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr_vals[str(g)] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            eod = round(tpr_vals[privileged] - tpr_vals[unprivileged], 4)
            aod = round(
                ((tpr_vals[privileged] - tpr_vals[unprivileged]) +
                 (fpr_vals[privileged] - fpr_vals[unprivileged])) / 2,
                4,
            )
        else:
            spd = 0.0
            di = None
            eod = None
            aod = None

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "spd": spd,
            "di": di,
            "eod": eod,
            "aod": aod,
            "positive_rates_per_group": pos_rates,
        }

    # ── Feature preparation ───────────────────────────────────────────────────

    def _prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, y, sensitive) arrays ready for sklearn.

        - All numeric columns (excluding target) are used as features.
        - Categorical columns are label-encoded.
        - Rows with NaN in any feature are dropped.
        """
        feature_cols = [c for c in df.columns if c != target_col]
        work = df[feature_cols + [target_col]].copy()

        # Encode categoricals
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(work[col]):
                le = LabelEncoder()
                work[col] = le.fit_transform(work[col].astype(str).fillna("__missing__"))

        work = work.dropna()

        # Sensitive array (keep original encoded values for grouping)
        sensitive_encoded = work[sensitive_attr].values

        # Feature matrix: numeric features only (already encoded)
        X = work[feature_cols].select_dtypes(include="number").fillna(0).values
        y = work[target_col].astype(int).values

        return X, y, sensitive_encoded

    # ── Winner selection ──────────────────────────────────────────────────────

    @staticmethod
    def _pick_winner(rew: dict, thr: dict) -> str:
        """
        Choose the better strategy:
        - Prefer the one with the greater bias reduction (SPD drop).
        - If both achieve similar bias reduction (within 5%), prefer the one
          with higher accuracy retained.
        """
        rew_eff = rew.get("effects", {})
        thr_eff = thr.get("effects", {})

        rew_bias = rew_eff.get("bias_reduction_pct", 0) or 0
        thr_bias = thr_eff.get("bias_reduction_pct", 0) or 0
        rew_acc = rew_eff.get("accuracy_retained_pct", 100) or 100
        thr_acc = thr_eff.get("accuracy_retained_pct", 100) or 100

        if abs(rew_bias - thr_bias) <= 5.0:
            # Similar bias reduction → prefer higher accuracy
            return "reweigh" if rew_acc >= thr_acc else "threshold"
        return "reweigh" if rew_bias >= thr_bias else "threshold"
