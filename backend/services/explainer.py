"""
FairLens — Bias Explainer Service

Analyses *why* bias exists in a dataset:
  - Correlation between the sensitive attribute and target
  - Proxy feature detection (non-sensitive columns correlated with the attribute)
  - Data imbalance across demographic groups
  - Historical skew (variance in positive rates)
  - A plain-English reason string combining all findings
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class BiasExplainer:
    """Explain the root causes of bias for a single sensitive attribute."""

    TOP_PROXY_N: int = 3
    IMBALANCE_THRESHOLD: float = 0.5

    def explain(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> dict[str, Any]:
        """
        Run all explanation analyses and return a unified dict.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
        sensitive_attr : str

        Returns
        -------
        dict with keys:
            correlation, proxy_features, data_imbalance,
            historical_skew, positive_rate_gap, plain_reason
        """
        work = df.copy().dropna(subset=[target_col, sensitive_attr])

        # ── 1. Correlation: sensitive_attr ↔ target ───────────────────────────
        correlation = self._correlation(work, sensitive_attr, target_col)

        # ── 2. Proxy features ─────────────────────────────────────────────────
        proxy_features = self._proxy_features(work, target_col, sensitive_attr)

        # ── 3. Data imbalance ─────────────────────────────────────────────────
        group_counts = work[sensitive_attr].value_counts()
        smallest = int(group_counts.min())
        largest = int(group_counts.max())
        imbalance_ratio = round(smallest / largest, 4) if largest > 0 else 1.0
        imbalance_flagged = imbalance_ratio < self.IMBALANCE_THRESHOLD

        data_imbalance = {
            "ratio": imbalance_ratio,
            "smallest_group_count": smallest,
            "largest_group_count": largest,
            "flagged": imbalance_flagged,
            "interpretation": (
                f"The smallest group has only {imbalance_ratio:.0%} as many samples "
                "as the largest group. This may cause the model to underfit minority groups."
                if imbalance_flagged
                else "Group sizes are reasonably balanced."
            ),
        }

        # ── 4. Historical skew (std of positive rates across groups) ──────────
        positive_rates = (
            work.groupby(sensitive_attr)[target_col]
            .mean()
            .dropna()
        )
        historical_skew = round(float(positive_rates.std()), 4) if len(positive_rates) > 1 else 0.0

        # ── 5. Positive rate gap ──────────────────────────────────────────────
        positive_rate_gap = round(
            float(positive_rates.max() - positive_rates.min()), 4
        ) if len(positive_rates) > 1 else 0.0

        # ── 6. Plain-English reason ───────────────────────────────────────────
        plain_reason = self._build_plain_reason(
            sensitive_attr=sensitive_attr,
            target_col=target_col,
            correlation=correlation,
            proxy_features=proxy_features,
            positive_rate_gap=positive_rate_gap,
            imbalance_flagged=imbalance_flagged,
            historical_skew=historical_skew,
        )

        return {
            "sensitive_attr": sensitive_attr,
            "target_col": target_col,
            "correlation": round(correlation, 4),
            "proxy_features": proxy_features,
            "data_imbalance": data_imbalance,
            "historical_skew": historical_skew,
            "positive_rate_gap": positive_rate_gap,
            "plain_reason": plain_reason,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _correlation(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
    ) -> float:
        """
        Return Pearson correlation between two columns.
        Categorical columns are label-encoded first.
        """
        a = self._to_numeric(df[col_a])
        b = self._to_numeric(df[col_b])
        corr = a.corr(b)
        return float(abs(corr)) if not np.isnan(corr) else 0.0

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna("__missing__"))
        return pd.Series(encoded, index=series.index, dtype=float)

    def _proxy_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attr: str,
    ) -> list[dict[str, Any]]:
        """
        Find the top-N non-target, non-sensitive columns most correlated
        with the sensitive attribute.
        """
        sensitive_numeric = self._to_numeric(df[sensitive_attr])
        candidates = [
            c for c in df.columns
            if c not in (target_col, sensitive_attr)
        ]

        scores: list[tuple[str, float]] = []
        for col in candidates:
            try:
                col_numeric = self._to_numeric(df[col].dropna())
                # Align indices
                aligned_sensitive = sensitive_numeric.loc[col_numeric.index]
                corr = float(abs(aligned_sensitive.corr(col_numeric)))
                if not np.isnan(corr):
                    scores.append((col, corr))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: self.TOP_PROXY_N]

        result = []
        for feature, corr in top:
            if corr >= 0.5:
                strength = "strong"
                interp = (
                    f"'{feature}' is strongly correlated (r={corr:.2f}) with "
                    f"'{sensitive_attr}' and may act as a proxy variable, "
                    "allowing the model to discriminate indirectly."
                )
            elif corr >= 0.3:
                strength = "moderate"
                interp = (
                    f"'{feature}' has a moderate correlation (r={corr:.2f}) with "
                    f"'{sensitive_attr}'. Monitor for indirect discrimination."
                )
            else:
                strength = "weak"
                interp = (
                    f"'{feature}' has a weak correlation (r={corr:.2f}) with "
                    f"'{sensitive_attr}'. Unlikely to be a significant proxy."
                )

            result.append(
                {
                    "feature": feature,
                    "correlation": round(corr, 4),
                    "strength": strength,
                    "interpretation": interp,
                }
            )

        return result

    def _build_plain_reason(
        self,
        sensitive_attr: str,
        target_col: str,
        correlation: float,
        proxy_features: list[dict],
        positive_rate_gap: float,
        imbalance_flagged: bool,
        historical_skew: float,
    ) -> str:
        """Synthesise a plain-English explanation from all computed signals."""
        parts: list[str] = []

        # Proxy feature lead sentence
        if proxy_features and proxy_features[0]["correlation"] >= 0.3:
            top = proxy_features[0]
            parts.append(
                f"'{top['feature']}' is {top['strength']}ly correlated with "
                f"'{sensitive_attr}' (r={top['correlation']:.2f}), meaning the model "
                "may use it as a proxy for the protected attribute."
            )

        # Outcome gap
        if positive_rate_gap > 0.05:
            parts.append(
                f"There is a {positive_rate_gap:.0%} outcome gap between demographic "
                f"groups in '{target_col}', suggesting groups are treated differently."
            )

        # Data imbalance
        if imbalance_flagged:
            parts.append(
                f"The training data is imbalanced across groups of '{sensitive_attr}', "
                "which reduces the model's ability to learn fair representations for "
                "smaller groups."
            )

        # Direct correlation
        if correlation >= 0.3:
            parts.append(
                f"'{sensitive_attr}' itself has a notable direct correlation "
                f"(r={correlation:.2f}) with the target '{target_col}', "
                "indicating historical patterns of discrimination in the data."
            )

        # Historical skew
        if historical_skew > 0.1:
            parts.append(
                "High variance in positive rates across groups suggests this dataset "
                "reflects historical inequities that the model has learned and amplified."
            )

        if not parts:
            return (
                f"No strong bias signals were detected for '{sensitive_attr}'. "
                "The observed metrics may reflect natural variation rather than systematic bias. "
                "Continue to monitor with larger datasets."
            )

        return " ".join(parts)
