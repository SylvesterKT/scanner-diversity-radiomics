"""Scanner Profiler - Characterises scanner/site-specific radiomic feature distributions.

Profiles each scanner group's feature statistics, detects outlier scanners,
and quantifies inter-scanner variability to prioritise harmonisation efforts.

Author: Sylvester KT
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings


class ScannerProfiler:
    """Profile radiomic feature distributions across scanner/acquisition sites."""

    def __init__(self, distance_metric: str = 'js_divergence'):
        """
        Initialise ScannerProfiler.

        Parameters
        ----------
        distance_metric : str
            Inter-scanner distance metric. Options: 'js_divergence', 'wasserstein', 'ks_statistic'.
        """
        self.distance_metric = distance_metric
        self.scanner_profiles_: Optional[Dict[str, pd.DataFrame]] = None
        self.inter_scanner_distances_: Optional[pd.DataFrame] = None
        self.feature_variability_: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(
        self,
        features_df: pd.DataFrame,
        scanner_labels: pd.Series
    ) -> 'ScannerProfiler':
        """
        Profile feature distributions for each scanner.

        Parameters
        ----------
        features_df : pd.DataFrame
            Shape (n_samples, n_features)
        scanner_labels : pd.Series
            Scanner identifier per sample

        Returns
        -------
        self
        """
        self.scanner_profiles_ = {}
        unique_scanners = scanner_labels.unique()

        for scanner in unique_scanners:
            mask = scanner_labels == scanner
            site_data = features_df[mask]
            profile = pd.DataFrame({
                'mean': site_data.mean(),
                'std': site_data.std(ddof=1),
                'median': site_data.median(),
                'iqr': site_data.quantile(0.75) - site_data.quantile(0.25),
                'skewness': site_data.skew(),
                'kurtosis': site_data.kurtosis(),
                'min': site_data.min(),
                'max': site_data.max(),
                'n_samples': mask.sum()
            })
            self.scanner_profiles_[scanner] = profile

        # Compute inter-scanner distances
        self.inter_scanner_distances_ = self._compute_inter_scanner_distances(
            features_df, scanner_labels, unique_scanners
        )

        # Compute per-feature variability across scanners
        self.feature_variability_ = self._compute_feature_variability(unique_scanners)

        self._fitted = True
        return self

    def _compute_inter_scanner_distances(
        self,
        features_df: pd.DataFrame,
        scanner_labels: pd.Series,
        unique_scanners: np.ndarray
    ) -> pd.DataFrame:
        """Compute pairwise inter-scanner distances averaged across features."""
        n = len(unique_scanners)
        dist_matrix = np.zeros((n, n))

        for i, s1 in enumerate(unique_scanners):
            for j, s2 in enumerate(unique_scanners):
                if i >= j:
                    continue
                d = self._pairwise_distance(
                    features_df[scanner_labels == s1],
                    features_df[scanner_labels == s2]
                )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return pd.DataFrame(
            dist_matrix,
            index=unique_scanners,
            columns=unique_scanners
        )

    def _pairwise_distance(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ) -> float:
        """Compute mean feature distance between two scanner groups."""
        distances = []
        for col in data1.columns:
            x = data1[col].dropna().values
            y = data2[col].dropna().values
            if len(x) < 2 or len(y) < 2:
                continue

            try:
                if self.distance_metric == 'js_divergence':
                    # Compute JS divergence via histograms
                    bins = np.linspace(
                        min(x.min(), y.min()),
                        max(x.max(), y.max()),
                        30
                    )
                    p, _ = np.histogram(x, bins=bins, density=True)
                    q, _ = np.histogram(y, bins=bins, density=True)
                    # Add small epsilon to avoid log(0)
                    p = p + 1e-10
                    q = q + 1e-10
                    p /= p.sum()
                    q /= q.sum()
                    d = float(jensenshannon(p, q))

                elif self.distance_metric == 'wasserstein':
                    d = float(stats.wasserstein_distance(x, y))

                elif self.distance_metric == 'ks_statistic':
                    ks_result = stats.ks_2samp(x, y)
                    d = float(ks_result.statistic)

                else:
                    raise ValueError(f"Unknown distance metric: {self.distance_metric}")

                distances.append(d)
            except Exception as e:
                warnings.warn(f"Distance computation failed for feature: {e}")
                continue

        return float(np.mean(distances)) if distances else np.nan

    def _compute_feature_variability(self, unique_scanners: np.ndarray) -> pd.DataFrame:
        """Compute coefficient of variation of feature means across scanners."""
        means_across_scanners = pd.DataFrame({
            s: self.scanner_profiles_[s]['mean']
            for s in unique_scanners
        })

        cv = means_across_scanners.std(axis=1) / (means_across_scanners.mean(axis=1).abs() + 1e-10)
        range_ratio = (
            (means_across_scanners.max(axis=1) - means_across_scanners.min(axis=1))
            / (means_across_scanners.mean(axis=1).abs() + 1e-10)
        )

        return pd.DataFrame({
            'cv_across_scanners': cv,
            'range_ratio': range_ratio,
            'mean_across_scanners': means_across_scanners.mean(axis=1),
            'std_across_scanners': means_across_scanners.std(axis=1)
        }).sort_values('cv_across_scanners', ascending=False)

    def get_most_variable_features(self, n: int = 20) -> List[str]:
        """Return the n most scanner-variable features (highest CV)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_most_variable_features().")
        return list(self.feature_variability_.head(n).index)

    def get_least_variable_features(self, n: int = 20) -> List[str]:
        """Return the n least scanner-variable features (lowest CV)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_least_variable_features().")
        return list(self.feature_variability_.tail(n).index)

    def get_outlier_scanners(
        self,
        threshold_percentile: float = 90.0
    ) -> List[str]:
        """
        Identify scanner sites that are outliers (high mean inter-scanner distance).

        Parameters
        ----------
        threshold_percentile : float
            Percentile above which a scanner is considered an outlier.

        Returns
        -------
        List of scanner IDs classified as outliers.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before get_outlier_scanners().")

        mean_distances = self.inter_scanner_distances_.mean(axis=1)
        threshold = np.percentile(mean_distances.values, threshold_percentile)
        return list(mean_distances[mean_distances > threshold].index)

    def summarise(self) -> pd.DataFrame:
        """Return a summary table of all scanner profiles."""
        if not self._fitted:
            raise RuntimeError("Call fit() before summarise().")

        rows = []
        for scanner, profile in self.scanner_profiles_.items():
            rows.append({
                'scanner': scanner,
                'n_samples': int(profile['n_samples'].iloc[0]),
                'mean_feature_mean': float(profile['mean'].mean()),
                'mean_feature_std': float(profile['std'].mean()),
                'mean_skewness': float(profile['skewness'].mean()),
                'features_profiled': len(profile)
            })
        return pd.DataFrame(rows).set_index('scanner')

    def feature_anova_test(
        self,
        features_df: pd.DataFrame,
        scanner_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Run one-way ANOVA for each feature across scanners.
        Identifies features with significant scanner-group differences.

        Returns
        -------
        pd.DataFrame with columns: feature, f_statistic, p_value, significant
        """
        results = []
        for feature in features_df.columns:
            groups = [
                features_df[feature][scanner_labels == s].dropna().values
                for s in scanner_labels.unique()
            ]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 2:
                continue
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                results.append({
                    'feature': feature,
                    'f_statistic': float(f_stat),
                    'p_value': float(p_val),
                    'significant_p05': p_val < 0.05,
                    'significant_bonferroni': p_val < (0.05 / len(features_df.columns))
                })
            except Exception as e:
                warnings.warn(f"ANOVA failed for {feature}: {e}")

        return pd.DataFrame(results).set_index('feature').sort_values('p_value')
