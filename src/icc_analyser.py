"""ICC Analyser - Intraclass Correlation Coefficient computation for radiomic features across scanner groups.

Computes ICC(2,1) for each radiomic feature to assess cross-scanner reproducibility.
Features with ICC > threshold are considered scanner-stable.

Author: Sylvester KT
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings


class ICCAnalyser:
    """Compute Intraclass Correlation Coefficients for radiomic feature stability assessment."""

    def __init__(self, icc_threshold: float = 0.75, icc_type: str = 'ICC(2,1)'):
        """
        Initialise ICCAnalyser.

        Parameters
        ----------
        icc_threshold : float
            Minimum ICC value for a feature to be considered stable (default 0.75)
        icc_type : str
            ICC type to compute. Currently supports ICC(2,1) - two-way random, single measures.
        """
        self.icc_threshold = icc_threshold
        self.icc_type = icc_type
        self.icc_results_: Optional[pd.DataFrame] = None
        self.stable_features_: Optional[List[str]] = None
        self.unstable_features_: Optional[List[str]] = None

    def compute(self, features_df: pd.DataFrame, scanner_labels: pd.Series) -> pd.DataFrame:
        """
        Compute ICC for all features across scanner groups.

        Parameters
        ----------
        features_df : pd.DataFrame
            Shape (n_samples, n_features). Each column is a radiomic feature.
        scanner_labels : pd.Series
            Scanner/site identifier for each sample. Same index as features_df.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, icc, lower_ci, upper_ci, p_value, stable
        """
        results = []
        unique_scanners = scanner_labels.unique()

        if len(unique_scanners) < 2:
            raise ValueError("At least 2 scanner groups are required for ICC computation.")

        for feature in features_df.columns:
            try:
                icc_val, lower_ci, upper_ci, p_val = self._compute_icc21(
                    features_df[feature], scanner_labels
                )
                results.append({
                    'feature': feature,
                    'icc': icc_val,
                    'lower_ci_95': lower_ci,
                    'upper_ci_95': upper_ci,
                    'p_value': p_val,
                    'stable': icc_val >= self.icc_threshold
                })
            except Exception as e:
                warnings.warn(f"Could not compute ICC for {feature}: {e}")
                results.append({
                    'feature': feature,
                    'icc': np.nan,
                    'lower_ci_95': np.nan,
                    'upper_ci_95': np.nan,
                    'p_value': np.nan,
                    'stable': False
                })

        self.icc_results_ = pd.DataFrame(results).set_index('feature')
        stable_mask = self.icc_results_['stable']
        self.stable_features_ = list(self.icc_results_[stable_mask].index)
        self.unstable_features_ = list(self.icc_results_[~stable_mask].index)
        return self.icc_results_

    def _compute_icc21(
        self,
        feature_values: pd.Series,
        scanner_labels: pd.Series
    ) -> Tuple[float, float, float, float]:
        """
        Compute ICC(2,1): two-way random effects, single measures.

        Uses one-way ANOVA approach suitable for unbalanced designs.

        Returns
        -------
        Tuple of (icc, lower_ci, upper_ci, p_value)
        """
        groups = [feature_values[scanner_labels == s].values
                  for s in scanner_labels.unique()]

        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(*groups)

        # ANOVA components
        all_vals = feature_values.values
        grand_mean = np.mean(all_vals)
        n_total = len(all_vals)
        k = len(groups)
        n_j = np.array([len(g) for g in groups])

        # Mean squares
        ss_between = sum(n_j[i] * (np.mean(groups[i]) - grand_mean) ** 2 for i in range(k))
        ss_within = sum(np.sum((groups[i] - np.mean(groups[i])) ** 2) for i in range(k))
        df_between = k - 1
        df_within = n_total - k

        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 1e-10

        # Harmonic mean of group sizes
        n0 = (n_total - np.sum(n_j ** 2) / n_total) / (k - 1)

        # ICC(2,1) estimate (simplified without rater variance - suitable for scanner groups)
        ms_error = ms_within
        icc = (ms_between - ms_error) / (ms_between + (n0 - 1) * ms_error)
        icc = float(np.clip(icc, -1.0, 1.0))

        # 95% CI using F-distribution
        alpha = 0.05
        f_lower = f_stat / stats.f.ppf(1 - alpha / 2, df_between, df_within)
        f_upper = f_stat / stats.f.ppf(alpha / 2, df_between, df_within)
        lower_ci = (f_lower - 1) / (f_lower + n0 - 1)
        upper_ci = (f_upper - 1) / (f_upper + n0 - 1)

        return icc, float(lower_ci), float(upper_ci), float(p_val)

    def get_stable_features(self) -> List[str]:
        """Return list of features with ICC >= threshold."""
        if self.stable_features_ is None:
            raise RuntimeError("Call compute() before get_stable_features().")
        return self.stable_features_

    def get_unstable_features(self) -> List[str]:
        """Return list of features with ICC < threshold."""
        if self.unstable_features_ is None:
            raise RuntimeError("Call compute() before get_unstable_features().")
        return self.unstable_features_

    def summarise(self) -> Dict[str, float]:
        """Return summary statistics of ICC distribution."""
        if self.icc_results_ is None:
            raise RuntimeError("Call compute() before summarise().")
        icc_vals = self.icc_results_['icc'].dropna()
        return {
            'n_features': len(icc_vals),
            'n_stable': len(self.stable_features_),
            'n_unstable': len(self.unstable_features_),
            'pct_stable': 100 * len(self.stable_features_) / len(icc_vals),
            'mean_icc': float(icc_vals.mean()),
            'median_icc': float(icc_vals.median()),
            'std_icc': float(icc_vals.std()),
            'min_icc': float(icc_vals.min()),
            'max_icc': float(icc_vals.max()),
        }

    def feature_class_breakdown(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Break down ICC stability by PyRadiomics feature class.

        Assumes feature names follow PyRadiomics naming: original_<class>_<name>
        """
        if self.icc_results_ is None:
            raise RuntimeError("Call compute() before feature_class_breakdown().")

        df = self.icc_results_.copy().reset_index()
        df['feature_class'] = df['feature'].apply(self._extract_feature_class)

        breakdown = df.groupby('feature_class').agg(
            n_features=('icc', 'count'),
            n_stable=('stable', 'sum'),
            mean_icc=('icc', 'mean'),
            median_icc=('icc', 'median')
        ).reset_index()
        breakdown['pct_stable'] = 100 * breakdown['n_stable'] / breakdown['n_features']
        return breakdown.sort_values('pct_stable', ascending=False)

    @staticmethod
    def _extract_feature_class(feature_name: str) -> str:
        """Extract PyRadiomics feature class from feature name."""
        parts = feature_name.split('_')
        if len(parts) >= 2:
            return parts[1] if parts[0] in ('original', 'wavelet', 'log') else 'unknown'
        return 'unknown'
