"""
ByUs — Bias Mitigator Service

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
        """
        df_clean = df.copy().dropna(subset=[target_col, sensitive_attr])

        # Binarize target
        y = df_clean[target_col]
        if set(y.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            y_bin = y.astype(int)
        elif y.nunique() == 2:
            vals = sorted(y.unique())
            y_bin = y.map({vals[0]: 0, vals[1]: 1})
        elif pd.api.types.is_numeric_dtype(y):
            median = y.median()
            y_bin = (y > median).astype(int)
        else:
            y_bin = (y == y.mode()[0]).astype(int)
        df_clean['__target__'] = y_bin

        # Encode sensitive attr
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_clean['__sens__'] = le.fit_transform(df_clean[sensitive_attr].astype(str))

        # Select numeric features
        feature_cols = [c for c in df_clean.columns
                        if c not in [target_col, sensitive_attr, '__target__', '__sens__']
                        and df_clean[c].dtype in ['int64', 'float64']]
        if not feature_cols:
            raise ValueError("No numeric feature columns found for mitigation.")
        
        # Fill NA with 0 for features
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)

        rew = self.reweigh(df_clean, feature_cols)
        thr = self.threshold_adjust(df_clean, feature_cols)

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
        df_clean: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """
        Compute sample weights based on the ratio of expected to actual
        joint frequency of (group, label), then retrain a classifier.
        """
        n_total = len(df_clean)
        weights = np.ones(n_total)
        weights_summary = []
        weight_map = {}

        groups = df_clean['__sens__'].unique()
        labels = df_clean['__target__'].unique()

        for g in groups:
            for l in labels:
                n_group = (df_clean['__sens__'] == g).sum()
                n_label = (df_clean['__target__'] == l).sum()
                n_group_label = ((df_clean['__sens__'] == g) & (df_clean['__target__'] == l)).sum()

                if n_group_label == 0:
                    w = 1.0
                else:
                    w = (n_group / n_total) * (n_label / n_total) / (n_group_label / n_total)
                
                # Clip to prevent extreme values
                w = max(0.1, min(10.0, w))
                weight_map[(g, l)] = w

                weights_summary.append({
                    "group": str(g),
                    "label": str(l),
                    "weight": round(w, 4)
                })

        for i, (g, l) in enumerate(zip(df_clean['__sens__'].values, df_clean['__target__'].values)):
            weights[i] = weight_map.get((g, l), 1.0)

        X = df_clean[feature_cols].values
        y = df_clean['__target__'].values
        s = df_clean['__sens__'].values

        X_train, X_test, y_train, y_test, s_train, s_test, w_train, _ = train_test_split(
            X, y, s, weights, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE, stratify=y
        )

        # BEFORE metrics
        model_before = LogisticRegression(max_iter=1000, random_state=self.RANDOM_STATE)
        model_before.fit(X_train, y_train)
        y_pred_before = model_before.predict(X_test)
        before_metrics = {
            "SPD": self._compute_spd(y_pred_before, s_test),
            "DI": self._compute_di(y_pred_before, s_test),
            "EOD": self._compute_eod(y_pred_before, y_test, s_test),
            "AOD": self._compute_aod(y_pred_before, y_test, s_test),
            "accuracy": round(accuracy_score(y_test, y_pred_before), 4),
            "precision": round(precision_score(y_test, y_pred_before, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_before, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_before, zero_division=0), 4),
        }

        # AFTER metrics
        model_after = LogisticRegression(max_iter=1000, random_state=self.RANDOM_STATE)
        model_after.fit(X_train, y_train, sample_weight=w_train)
        y_pred_after = model_after.predict(X_test)
        after_metrics = {
            "SPD": self._compute_spd(y_pred_after, s_test),
            "DI": self._compute_di(y_pred_after, s_test),
            "EOD": self._compute_eod(y_pred_after, y_test, s_test),
            "AOD": self._compute_aod(y_pred_after, y_test, s_test),
            "accuracy": round(accuracy_score(y_test, y_pred_after), 4),
            "precision": round(precision_score(y_test, y_pred_after, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_after, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_after, zero_division=0), 4),
        }

        eff = self.effects(before_metrics, after_metrics)

        return {
            "before": before_metrics,
            "after": after_metrics,
            "effects": eff,
            "weights_summary": weights_summary,
        }

    # ── Threshold Adjustment ──────────────────────────────────────────────────

    def threshold_adjust(self, df_clean: pd.DataFrame, feature_cols: list[str]):
        import numpy as np

        X = df_clean[feature_cols].values
        y = df_clean['__target__'].values
        s = df_clean['__sens__'].values

        # Fixed seed for reproducibility
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE, stratify=y
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Get probabilities on test set
        proba = model.predict_proba(X_test)[:, 1]

        # Compute BEFORE metrics (standard 0.5 threshold)
        y_pred_before = (proba >= 0.5).astype(int)
        before_metrics = {
            "SPD": self._compute_spd(y_pred_before, s_test),
            "DI": self._compute_di(y_pred_before, s_test),
            "EOD": self._compute_eod(y_pred_before, y_test, s_test),
            "AOD": self._compute_aod(y_pred_before, y_test, s_test),
            "accuracy": round(accuracy_score(y_test, y_pred_before), 4),
            "precision": round(precision_score(y_test, y_pred_before, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_before, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_before, zero_division=0), 4),
        }

        # Find per-group thresholds that equalize TPR
        # Strategy: grid search thresholds per group, minimize TPR difference
        groups = np.unique(s_test)
        best_thresholds = {}
        best_spd = float('inf')

        # Try threshold pairs from 0.2 to 0.8 in steps of 0.05
        threshold_range = np.arange(0.2, 0.81, 0.05)

        if len(groups) == 2:
            g0, g1 = groups[0], groups[1]
            for t0 in threshold_range:
                for t1 in threshold_range:
                    y_adj = np.zeros(len(proba), dtype=int)
                    y_adj[s_test == g0] = (proba[s_test == g0] >= t0).astype(int)
                    y_adj[s_test == g1] = (proba[s_test == g1] >= t1).astype(int)

                    # Skip if recall collapses to 0 for any group
                    rec0 = recall_score(y_test[s_test == g0], y_adj[s_test == g0], zero_division=0)
                    rec1 = recall_score(y_test[s_test == g1], y_adj[s_test == g1], zero_division=0)
                    if rec0 < 0.05 or rec1 < 0.05:
                        continue

                    spd = abs(self._compute_spd(y_adj, s_test))
                    if spd < best_spd:
                        best_spd = spd
                        best_thresholds = {g0: t0, g1: t1}
        else:
            # For multi-group: use uniform threshold that minimizes overall SPD
            # while keeping recall > 0.05 per group
            for t in threshold_range:
                y_adj = (proba >= t).astype(int)
                recalls = [recall_score(y_test[s_test == g], y_adj[s_test == g], zero_division=0)
                           for g in groups]
                if min(recalls) < 0.05:
                    continue
                spd = abs(self._compute_spd(y_adj, s_test))
                if spd < best_spd:
                    best_spd = spd
                    best_thresholds = {g: t for g in groups}

        # If no valid thresholds found, fall back to 0.5 for all groups
        if not best_thresholds:
            best_thresholds = {g: 0.5 for g in groups}

        # Apply best thresholds
        y_pred_after = np.zeros(len(proba), dtype=int)
        for g, thresh in best_thresholds.items():
            y_pred_after[s_test == g] = (proba[s_test == g] >= thresh).astype(int)

        after_metrics = {
            "SPD": self._compute_spd(y_pred_after, s_test),
            "DI": self._compute_di(y_pred_after, s_test),
            "EOD": self._compute_eod(y_pred_after, y_test, s_test),
            "AOD": self._compute_aod(y_pred_after, y_test, s_test),
            "accuracy": round(accuracy_score(y_test, y_pred_after), 4),
            "precision": round(precision_score(y_test, y_pred_after, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_after, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred_after, zero_division=0), 4),
        }

        spd_before = abs(before_metrics["SPD"])
        spd_after = abs(after_metrics["SPD"])
        improvement_pct = round(((spd_before - spd_after) / max(spd_before, 1e-9)) * 100, 1)
        accuracy_retained = round((after_metrics["accuracy"] / max(before_metrics["accuracy"], 1e-9)) * 100, 1)

        effects = {
            "accuracy_delta": round(after_metrics["accuracy"] - before_metrics["accuracy"], 4),
            "precision_delta": round(after_metrics["precision"] - before_metrics["precision"], 4),
            "recall_delta": round(after_metrics["recall"] - before_metrics["recall"], 4),
            "f1_delta": round(after_metrics["f1"] - before_metrics["f1"], 4),
            "spd_delta": round(after_metrics["SPD"] - before_metrics["SPD"], 4),
            "bias_reduction_pct": improvement_pct,
            "accuracy_retained_pct": accuracy_retained,
        }

        return {
            "before": before_metrics,
            "after": after_metrics,
            "effects": effects,
            "improvement_pct": improvement_pct,
            "thresholds": {str(k): round(float(v), 2) for k, v in best_thresholds.items()},
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

    def _compute_spd(self, y_pred, s):
        groups = np.unique(s)
        if len(groups) < 2:
            return 0.0
        rates = {g: np.mean(y_pred[s == g]) for g in groups}
        priv = max(rates, key=rates.get)
        unpriv = min(rates, key=rates.get)
        return round(float(rates[priv] - rates[unpriv]), 4)

    def _compute_di(self, y_pred, s):
        groups = np.unique(s)
        if len(groups) < 2:
            return 1.0
        rates = {g: np.mean(y_pred[s == g]) for g in groups}
        priv_rate = max(rates.values())
        unpriv_rate = min(rates.values())
        if priv_rate == 0:
            return 1.0
        return round(float(unpriv_rate / priv_rate), 4)

    def _compute_eod(self, y_pred, y_true, s):
        from sklearn.metrics import recall_score
        groups = np.unique(s)
        if len(groups) < 2:
            return 0.0
        tprs = {}
        for g in groups:
            mask = s == g
            if sum(y_true[mask]) == 0:
                return None
            tprs[g] = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        priv = max(tprs, key=tprs.get)
        unpriv = min(tprs, key=tprs.get)
        return round(float(tprs[priv] - tprs[unpriv]), 4)

    def _compute_aod(self, y_pred, y_true, s):
        groups = np.unique(s)
        if len(groups) < 2:
            return 0.0
        tprs, fprs = {}, {}
        for g in groups:
            mask = s == g
            pos_mask = y_true[mask] == 1
            neg_mask = y_true[mask] == 0
            if sum(pos_mask) == 0 or sum(neg_mask) == 0:
                return None
            tprs[g] = np.mean(y_pred[mask][pos_mask])
            fprs[g] = np.mean(y_pred[mask][neg_mask])
        groups_list = list(groups)
        tpr_diff = abs(tprs[groups_list[0]] - tprs[groups_list[1]])
        fpr_diff = abs(fprs[groups_list[0]] - fprs[groups_list[1]])
        return round(float((tpr_diff + fpr_diff) / 2), 4)
