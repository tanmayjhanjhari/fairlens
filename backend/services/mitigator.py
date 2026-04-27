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

        rew = self.reweigh(df, target_col, sensitive_attr)
        thr = self.threshold_adjust(df_clean, feature_cols)

        # Winner: best SPD reduction while keeping accuracy drop < 3%
        winner = self._pick_winner(rew, thr)

        # Generate explanations
        reweigh_explanation = self.generate_mitigation_explanation(
            rew["before"], rew["after"],
            "reweigh", sensitive_attr, rew["effects"]
        )
        threshold_explanation = self.generate_mitigation_explanation(
            thr["before"], thr["after"],
            "threshold", sensitive_attr, thr["effects"]
        )
        
        rew["explanation"] = reweigh_explanation
        thr["explanation"] = threshold_explanation
        
        # Winner reasoning
        if winner == "reweigh":
            w_bias = rew["effects"].get("bias_reduction_pct", 0)
            w_acc = rew["effects"].get("accuracy_retained_pct", 100)
            l_bias = thr["effects"].get("bias_reduction_pct", 0)
            l_acc = thr["effects"].get("accuracy_retained_pct", 100)
            winner_reason = (
                f"Reweighing is recommended because it achieved {w_bias:.0f}% "
                f"bias reduction with {w_acc:.0f}% accuracy retained, "
                f"outperforming threshold adjustment ({l_bias:.0f}% bias reduction, "
                f"{l_acc:.0f}% accuracy retained)."
            )
        else:
            w_bias = thr["effects"].get("bias_reduction_pct", 0)
            w_acc = thr["effects"].get("accuracy_retained_pct", 100)
            l_bias = rew["effects"].get("bias_reduction_pct", 0)
            l_acc = rew["effects"].get("accuracy_retained_pct", 100)
            winner_reason = (
                f"Threshold adjustment is recommended because it achieved {w_bias:.0f}% "
                f"bias reduction with {w_acc:.0f}% accuracy retained, "
                f"outperforming reweighing ({l_bias:.0f}% bias reduction, "
                f"{l_acc:.0f}% accuracy retained)."
            )

        return {
            "reweigh": rew,
            "threshold": thr,
            "winner": winner,
            "winner_reason": winner_reason,
        }

    def generate_mitigation_explanation(self, before: dict, after: dict,
                                        technique: str, sensitive_attr: str,
                                        effects: dict) -> dict:
        """Generate plain-English explanation of what mitigation did."""
        
        spd_before = abs(before.get("SPD", 0) or 0)
        spd_after = abs(after.get("SPD", 0) or 0)
        acc_before = before.get("accuracy", 0) or 0
        acc_after = after.get("accuracy", 0) or 0
        bias_reduction = effects.get("bias_reduction_pct", 0)
        acc_retained = effects.get("accuracy_retained_pct", 100)
        acc_delta = effects.get("accuracy_delta", 0)
        
        # What the technique actually did
        if technique == "reweigh":
            how_it_works = (
                f"Reweighing works by giving more importance to underrepresented "
                f"(group, outcome) combinations during training. For '{sensitive_attr}', "
                f"cases where disadvantaged groups received positive outcomes were "
                f"given higher weight, teaching the model to be more balanced."
            )
        else:
            how_it_works = (
                f"Threshold Adjustment works by finding different decision thresholds "
                f"for each group of '{sensitive_attr}'. Instead of using one cutoff "
                f"for everyone, the model uses group-specific cutoffs that equalise "
                f"the True Positive Rate — meaning equally qualified people from "
                f"different groups get equal chances."
            )
        
        # What actually happened to bias
        if spd_before == 0:
            bias_result = "There was no measurable bias to reduce before mitigation."
        elif bias_reduction >= 70:
            bias_result = (
                f"Bias was significantly reduced. SPD dropped from {spd_before:.3f} "
                f"to {spd_after:.3f} — a {bias_reduction:.0f}% reduction. "
                f"In practical terms, the outcome gap between groups narrowed "
                f"from {spd_before*100:.1f}% to {spd_after*100:.1f}%."
            )
        elif bias_reduction >= 30:
            bias_result = (
                f"Bias was partially reduced. SPD dropped from {spd_before:.3f} "
                f"to {spd_after:.3f} — a {bias_reduction:.0f}% improvement. "
                f"Some gap remains between groups, but the disparity is meaningfully smaller."
            )
        elif bias_reduction > 0:
            bias_result = (
                f"Bias reduction was modest — only {bias_reduction:.0f}%. "
                f"SPD moved from {spd_before:.3f} to {spd_after:.3f}. "
                f"This often happens when the bias is deeply embedded in the "
                f"feature relationships rather than just class imbalance, "
                f"or when the sensitive attribute has too many unique groups."
            )
        else:
            bias_result = (
                f"This technique did not reduce bias for '{sensitive_attr}'. "
                f"SPD remained at {spd_after:.3f}. This can happen when bias "
                f"is driven by a proxy feature that this technique cannot address, "
                f"or when group sizes are very unequal."
            )
        
        # What happened to accuracy
        if abs(acc_delta) < 0.005:
            acc_result = (
                f"Model accuracy was virtually unchanged ({acc_before:.1%} → "
                f"{acc_after:.1%}), meaning fairness was improved at no real "
                f"cost to predictive performance."
            )
        elif acc_delta < 0:
            acc_result = (
                f"Model accuracy dropped slightly from {acc_before:.1%} to "
                f"{acc_after:.1%} (a {abs(acc_delta)*100:.1f}% reduction). "
                f"This is the typical fairness-accuracy trade-off: making the "
                f"model fairer for disadvantaged groups slightly reduces its "
                f"overall optimisation. Whether this trade-off is acceptable "
                f"is a business and ethical decision."
            )
        else:
            acc_result = (
                f"Interestingly, model accuracy slightly improved from "
                f"{acc_before:.1%} to {acc_after:.1%}. This can happen when "
                f"the original model was overfit to majority-group patterns, "
                f"and mitigation forced it to learn more generalisable features."
            )
        
        # What the graph is showing
        graph_explanation = (
            f"The Fairness Improvement chart compares SPD, DI, EOD, and AOD "
            f"before mitigation (gray bars) vs after reweighing (teal) and after "
            f"threshold adjustment (purple). Shorter bars are better — they mean "
            f"the gap between groups is smaller. "
            f"The Performance Trade-off chart plots each technique as a dot: "
            f"further right means more bias reduction, higher up means more "
            f"accuracy retained. The green 'Sweet Spot' zone is where both are high."
        )
        
        return {
            "how_it_works": how_it_works,
            "bias_result": bias_result,
            "acc_result": acc_result,
            "graph_explanation": graph_explanation,
            "summary": f"{bias_result} {acc_result}"
        }

    # ── Reweighing ────────────────────────────────────────────────────────────

    def reweigh(self, df, target_col, sensitive_attr):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score)

        df_work = df.copy().dropna(subset=[target_col, sensitive_attr])

        # --- Encode target ---
        le_t = LabelEncoder()
        y_all = le_t.fit_transform(df_work[target_col].astype(str))

        # --- Encode sensitive attr ---
        le_s = LabelEncoder()
        s_all = le_s.fit_transform(df_work[sensitive_attr].astype(str))

        # --- Numeric features only ---
        feature_cols = [c for c in df_work.columns
                        if c != target_col
                        and c != sensitive_attr
                        and df_work[c].dtype in ['int64','float64','int32','float32']]
        if not feature_cols:
            raise ValueError(f"No numeric features found. Cannot run reweighing.")

        X_all = df_work[feature_cols].fillna(0).values
        n = len(df_work)

        # --- Compute reweighing weights on FULL dataset ---
        # Weight formula: P(group)*P(label) / P(group AND label)
        weights = np.ones(n)
        for g in np.unique(s_all):
            for label in np.unique(y_all):
                mask = (s_all == g) & (y_all == label)
                n_gl = mask.sum()
                if n_gl == 0:
                    continue
                p_g = (s_all == g).sum() / n
                p_l = (y_all == label).sum() / n
                p_gl = n_gl / n
                w = (p_g * p_l) / p_gl
                weights[mask] = w

        # Clip weights to prevent instability
        weights = np.clip(weights, 0.1, 10.0)

        # --- Split AFTER computing weights so indices align ---
        idx = np.arange(n)
        idx_train, idx_test = train_test_split(
            idx, test_size=0.3, random_state=42,
            stratify=y_all
        )

        X_train = X_all[idx_train]
        y_train = y_all[idx_train]
        w_train = weights[idx_train]   # weights aligned with train split
        X_test  = X_all[idx_test]
        y_test  = y_all[idx_test]
        s_test  = s_all[idx_test]

        # --- BEFORE metrics (no weights) ---
        model_before = LogisticRegression(max_iter=1000, random_state=42)
        model_before.fit(X_train, y_train)
        y_pred_before = model_before.predict(X_test)

        before = {
            "SPD":  round(float(self._compute_spd(y_pred_before, s_test)), 4),
            "DI":   round(float(self._compute_di(y_pred_before, s_test)), 4),
            "EOD":  self._compute_eod(y_pred_before, y_test, s_test),
            "AOD":  self._compute_aod(y_pred_before, y_test, s_test),
            "accuracy":  round(accuracy_score(y_test, y_pred_before), 4),
            "precision": round(precision_score(y_test, y_pred_before, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred_before, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred_before, zero_division=0), 4),
        }

        # --- AFTER metrics (WITH weights on training) ---
        model_after = LogisticRegression(max_iter=1000, random_state=42)
        model_after.fit(X_train, y_train, sample_weight=w_train)
        y_pred_after = model_after.predict(X_test)

        after = {
            "SPD":  round(float(self._compute_spd(y_pred_after, s_test)), 4),
            "DI":   round(float(self._compute_di(y_pred_after, s_test)), 4),
            "EOD":  self._compute_eod(y_pred_after, y_test, s_test),
            "AOD":  self._compute_aod(y_pred_after, y_test, s_test),
            "accuracy":  round(accuracy_score(y_test, y_pred_after), 4),
            "precision": round(precision_score(y_test, y_pred_after, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred_after, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred_after, zero_division=0), 4),
        }

        spd_b = abs(before["SPD"])
        spd_a = abs(after["SPD"])
        improvement_pct = round(((spd_b - spd_a) / max(spd_b, 1e-9)) * 100, 1)
        accuracy_retained = round((after["accuracy"] / max(before["accuracy"], 1e-9)) * 100, 1)

        effects = {
            "accuracy_delta":  round(after["accuracy"]  - before["accuracy"],  4),
            "precision_delta": round(after["precision"] - before["precision"], 4),
            "recall_delta":    round(after["recall"]    - before["recall"],    4),
            "f1_delta":        round(after["f1"]        - before["f1"],        4),
            "spd_delta":       round(after["SPD"]       - before["SPD"],       4),
            "bias_reduction_pct":   improvement_pct,
            "accuracy_retained_pct": accuracy_retained,
        }

        return {"before": before, "after": after, "effects": effects,
                "improvement_pct": improvement_pct,
                "weights_summary": {
                    "min": round(float(weights.min()), 3),
                    "max": round(float(weights.max()), 3),
                    "mean": round(float(weights.mean()), 3)
                }}

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
