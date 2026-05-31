"""ComBat Harmonisation Module for cross-scanner radiomic feature normalisation.

Implements ComBat (Johnson et al. 2007) and Z-score harmonisation strategies
to recover unstable radiomic features after scanner diversity analysis.

Author: Sylvester KT
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.preprocessing import StandardScaler
import warnings


class ComBatHarmoniser:
    """ComBat batch effect correction for radiomic features across scanner sites."""

    def __init__(self, parametric: bool = True, mean_only: bool = False):
        """
        Initialise ComBatHarmoniser.

        Parameters
        ----------
        parametric : bool
            Use parametric ComBat (assumes gamma distribution of batch effects).
        mean_only : bool
            If True, correct only mean (location) effects, not variance (scale) effects.
        """
        self.parametric = parametric
        self.mean_only = mean_only
        self.gamma_hat_: Optional[np.ndarray] = None  # batch location effects
        self.delta_hat_: Optional[np.ndarray] = None  # batch scale effects
        self.grand_mean_: Optional[np.ndarray] = None
        self.var_pooled_: Optional[np.ndarray] = None
        self.scanner_labels_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._fitted = False

    def fit(self, features_df: pd.DataFrame, scanner_labels: pd.Series) -> 'ComBatHarmoniser':
        """
        Fit ComBat model to estimate batch effects.

        Parameters
        ----------
        features_df : pd.DataFrame
            Shape (n_samples, n_features)
        scanner_labels : pd.Series
            Scanner/batch identifier per sample

        Returns
        -------
        self
        """
        X = features_df.values.T  # ComBat convention: features x samples
        self.feature_names_ = list(features_df.columns)
        batches = scanner_labels.values
        unique_batches = np.unique(batches)
        n_batches = len(unique_batches)
        n_features, n_samples = X.shape

        # Grand mean across all samples
        self.grand_mean_ = np.mean(X, axis=1)  # (n_features,)

        # Pooled variance
        self.var_pooled_ = np.var(X, axis=1, ddof=1)  # (n_features,)
        self.var_pooled_ = np.where(self.var_pooled_ < 1e-10, 1e-10, self.var_pooled_)

        # Standardise
        X_std = (X - self.grand_mean_[:, np.newaxis]) / np.sqrt(self.var_pooled_[:, np.newaxis])

        # Estimate batch effects (gamma: location, delta: scale)
        self.gamma_hat_ = np.zeros((n_features, n_batches))
        self.delta_hat_ = np.ones((n_features, n_batches))
        self.scanner_labels_ = unique_batches

        for i, batch in enumerate(unique_batches):
            mask = batches == batch
            X_batch = X_std[:, mask]
            self.gamma_hat_[:, i] = np.mean(X_batch, axis=1)
            if not self.mean_only:
                var_batch = np.var(X_batch, axis=1, ddof=1)
                self.delta_hat_[:, i] = np.where(var_batch < 1e-10, 1.0, var_batch)

        self._fitted = True
        return self

    def transform(self, features_df: pd.DataFrame, scanner_labels: pd.Series) -> pd.DataFrame:
        """
        Apply ComBat harmonisation to remove batch effects.

        Parameters
        ----------
        features_df : pd.DataFrame
        scanner_labels : pd.Series

        Returns
        -------
        pd.DataFrame
            Harmonised features, same shape as input
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        X = features_df.values.T  # (n_features, n_samples)
        batches = scanner_labels.values
        X_harmonised = np.copy(X)

        # Standardise using fitted grand mean and pooled variance
        X_std = (X - self.grand_mean_[:, np.newaxis]) / np.sqrt(self.var_pooled_[:, np.newaxis])

        for i, batch in enumerate(self.scanner_labels_):
            mask = batches == batch
            if not np.any(mask):
                warnings.warn(f"Batch '{batch}' not found in transform data.")
                continue

            gamma = self.gamma_hat_[:, i][:, np.newaxis]  # (n_features, 1)
            delta = self.delta_hat_[:, i][:, np.newaxis]

            # Remove batch effect
            X_corrected = (X_std[:, mask] - gamma) / np.sqrt(delta)

            # Rescale back to original scale
            X_harmonised[:, mask] = (
                X_corrected * np.sqrt(self.var_pooled_[:, np.newaxis])
                + self.grand_mean_[:, np.newaxis]
            )

        return pd.DataFrame(
            X_harmonised.T,
            index=features_df.index,
            columns=features_df.columns
        )

    def fit_transform(
        self,
        features_df: pd.DataFrame,
        scanner_labels: pd.Series
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(features_df, scanner_labels).transform(features_df, scanner_labels)


class ZScoreHarmoniser:
    """Z-score normalisation per scanner site - simple baseline harmonisation."""

    def __init__(self):
        self.site_stats_: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._fitted = False

    def fit(self, features_df: pd.DataFrame, scanner_labels: pd.Series) -> 'ZScoreHarmoniser':
        """Compute per-site mean and std for each feature."""
        self.site_stats_ = {}
        for site in scanner_labels.unique():
            mask = scanner_labels == site
            site_data = features_df[mask]
            self.site_stats_[site] = {
                'mean': site_data.mean().values,
                'std': site_data.std(ddof=1).values.clip(min=1e-10)
            }
        self.feature_names_ = list(features_df.columns)
        self._fitted = True
        return self

    def transform(self, features_df: pd.DataFrame, scanner_labels: pd.Series) -> pd.DataFrame:
        """Apply per-site Z-score normalisation."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        X_out = features_df.values.copy().astype(float)
        for site, stats in self.site_stats_.items():
            mask = (scanner_labels == site).values
            X_out[mask] = (X_out[mask] - stats['mean']) / stats['std']

        return pd.DataFrame(X_out, index=features_df.index, columns=features_df.columns)

    def fit_transform(
        self,
        features_df: pd.DataFrame,
        scanner_labels: pd.Series
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(features_df, scanner_labels).transform(features_df, scanner_labels)


class HistogramMatchingHarmoniser:
    """Histogram matching harmonisation - aligns feature distributions to a reference site."""

    def __init__(self, reference_site: Optional[str] = None, n_quantiles: int = 1000):
        """
        Parameters
        ----------
        reference_site : str or None
            Site to use as reference. If None, uses the site with the most samples.
        n_quantiles : int
            Number of quantile bins for matching.
        """
        self.reference_site = reference_site
        self.n_quantiles = n_quantiles
        self.ref_quantiles_: Optional[Dict[str, np.ndarray]] = None
        self.site_quantiles_: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._fitted = False

    def fit(self, features_df: pd.DataFrame, scanner_labels: pd.Series) -> 'HistogramMatchingHarmoniser':
        """Compute quantile maps for each site and the reference."""
        quantiles = np.linspace(0, 100, self.n_quantiles)

        if self.reference_site is None:
            counts = scanner_labels.value_counts()
            self.reference_site = counts.idxmax()

        ref_mask = scanner_labels == self.reference_site
        ref_data = features_df[ref_mask]
        self.ref_quantiles_ = {
            col: np.percentile(ref_data[col].dropna(), quantiles)
            for col in features_df.columns
        }

        self.site_quantiles_ = {}
        for site in scanner_labels.unique():
            if site == self.reference_site:
                continue
            mask = scanner_labels == site
            site_data = features_df[mask]
            self.site_quantiles_[site] = {
                col: np.percentile(site_data[col].dropna(), quantiles)
                for col in features_df.columns
            }

        self.feature_names_ = list(features_df.columns)
        self._fitted = True
        return self

    def transform(self, features_df: pd.DataFrame, scanner_labels: pd.Series) -> pd.DataFrame:
        """Apply histogram matching to align non-reference sites to reference."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        X_out = features_df.copy()
        for site, site_q in self.site_quantiles_.items():
            mask = scanner_labels == site
            for col in features_df.columns:
                original_vals = features_df.loc[mask, col].values
                # Map via quantile interpolation
                matched = np.interp(
                    original_vals,
                    site_q[col],
                    self.ref_quantiles_[col]
                )
                X_out.loc[mask, col] = matched

        return X_out

    def fit_transform(
        self,
        features_df: pd.DataFrame,
        scanner_labels: pd.Series
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(features_df, scanner_labels).transform(features_df, scanner_labels)
