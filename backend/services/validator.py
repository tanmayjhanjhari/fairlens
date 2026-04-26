"""
FairLens — Data Validation Service

Validates a DataFrame / target / sensitive attribute combination before any
fairness analysis runs.  Returns a structured dict that downstream routers
can surface directly to the frontend.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class DataValidator:
    """Validate a dataset for FairLens fairness analysis."""

    # ── Thresholds ────────────────────────────────────────────────────────────
    MIN_ROWS_RELIABLE: int = 200
    MIN_ROWS_ABSOLUTE: int = 50
    MAX_GROUPS_STABLE: int = 10
    MAX_MISSING_PCT: float = 5.0
    MIN_MINORITY_CLASS_PCT: float = 15.0

    def validate(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attrs: list[str],
    ) -> dict[str, Any]:
        """
        Run all validation checks and return a comprehensive result dict.

        Parameters
        ----------
        df : pd.DataFrame
            The full dataset (already loaded from the uploaded CSV).
        target_col : str
            Name of the column to treat as the prediction target.
        sensitive_attrs : list[str]
            Columns to treat as sensitive / protected attributes.

        Returns
        -------
        dict with keys:
            supported, fallback_needed, engine, target_type,
            warnings, row_count, recommendations
        """
        warnings: list[str] = []
        recommendations: list[str] = []

        row_count = len(df)

        # ── 1. Target type ────────────────────────────────────────────────────
        n_unique_target = df[target_col].nunique()
        if n_unique_target == 2:
            target_type = "binary"
        elif n_unique_target <= 10:
            target_type = "multiclass"
        else:
            target_type = "continuous"

        # ── 2. Row count ──────────────────────────────────────────────────────
        if row_count < self.MIN_ROWS_ABSOLUTE:
            warnings.append(
                f"Dataset has only {row_count} rows, which is below the "
                f"absolute minimum of {self.MIN_ROWS_ABSOLUTE}. "
                "Analysis results will not be meaningful."
            )
            recommendations.append(
                "Collect at least 500 rows per demographic group for reliable results."
            )
        elif row_count < self.MIN_ROWS_RELIABLE:
            warnings.append(
                f"Dataset has {row_count} rows. Results may be statistically "
                f"unreliable. Recommend 500+ rows."
            )
            recommendations.append(
                "Consider collecting more data before acting on these results."
            )

        # ── 3. Sensitive attribute groups ─────────────────────────────────────
        too_many_groups_attrs: list[str] = []
        for attr in sensitive_attrs:
            if attr not in df.columns:
                warnings.append(
                    f"Sensitive attribute '{attr}' not found in the dataset."
                )
                continue
            n_groups = df[attr].nunique()
            if n_groups > self.MAX_GROUPS_STABLE:
                too_many_groups_attrs.append(attr)
                warnings.append(
                    f"'{attr}' has {n_groups} unique groups (>{self.MAX_GROUPS_STABLE}). "
                    "Reweighing may be unstable with this many groups."
                )
                recommendations.append(
                    f"Consider binning '{attr}' into fewer categories "
                    "(e.g., age → age_group) for more stable analysis."
                )

        # ── 4. Missing values ─────────────────────────────────────────────────
        for attr in sensitive_attrs:
            if attr not in df.columns:
                continue
            missing_pct = df[attr].isna().mean() * 100
            if missing_pct > self.MAX_MISSING_PCT:
                warnings.append(
                    f"'{attr}' has {missing_pct:.1f}% missing values. "
                    "These rows will be excluded from group analysis."
                )
                recommendations.append(
                    f"Impute or remove missing values in '{attr}' "
                    "before running fairness analysis."
                )

        # ── 5. Class imbalance ────────────────────────────────────────────────
        if target_type in ("binary", "multiclass"):
            value_counts = df[target_col].value_counts(normalize=True) * 100
            minority_pct = float(value_counts.min())
            if minority_pct < self.MIN_MINORITY_CLASS_PCT:
                warnings.append(
                    f"Target is highly imbalanced ({minority_pct:.1f}% minority class). "
                    "Metrics such as accuracy and DI may be skewed."
                )
                recommendations.append(
                    "Apply class-balancing (oversampling / undersampling) or "
                    "use precision/recall rather than accuracy as the primary metric."
                )

        # ── 6 / 7 / 8. Derive support flags ──────────────────────────────────
        all_attrs_exist = all(a in df.columns for a in sensitive_attrs)
        all_attrs_stable = len(too_many_groups_attrs) == 0

        supported: bool = (
            target_type == "binary"
            and all_attrs_exist
            and all_attrs_stable
            and row_count >= self.MIN_ROWS_ABSOLUTE
        )

        fallback_needed: bool = target_type in ("multiclass", "continuous")

        if supported:
            engine = "fairlens"
        else:
            engine = "fairlearn_fallback"

        return {
            "supported": supported,
            "fallback_needed": fallback_needed,
            "engine": engine,
            "target_type": target_type,
            "warnings": warnings,
            "row_count": row_count,
            "recommendations": recommendations,
        }
