from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import boxcox, zscore, iqr


class Preprocessor(ABC):
    def __init__(
            self,
            perf_type: str = "abs",  # or "rel"
            abs_threshold: float = 0.8,
            rel_threshold: float = 0.05,
    ):
        self.perf_type = perf_type
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold

    @abstractmethod
    def preprocess(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ...


class PRELIM(Preprocessor):
    def __init__(
            self,
            perf_type: str = "abs",  # or "rel"
            abs_threshold: float = 0.8,
            rel_threshold: float = 0.05,
    ):
        super().__init__(
            perf_type=perf_type,
            abs_threshold=abs_threshold,
            rel_threshold=rel_threshold
        )

    def preprocess(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        y = target.copy().to_numpy()
        x = features.copy().to_numpy()
        if self.perf_type == "rel":
            best_algo = np.max(np.where(np.isnan(y), -np.inf, y), axis=1)
            y[y == 0] = np.finfo(float).eps
            y = 1 - (y / best_algo[:, None])

        params = {'median': np.nanmedian(x, axis=0), 'iq_range': iqr(x, axis=0, nan_policy='omit')}
        params['hi_bound'] = params['median'] + 5 * params['iq_range']
        params['lo_bound'] = params['median'] - 5 * params['iq_range']

        hi_mask = x > params['hi_bound'][None, :]
        lo_mask = x < params['lo_bound'][None, :]
        x = np.where(hi_mask, params['hi_bound'][None, :], x)
        x = np.where(lo_mask, params['lo_bound'][None, :], x)

        features_to_remove = []
        features_names = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            if np.all(x_i == x_i[0]):
                features_to_remove.append(i)
                features_names.append(features.columns[i])

        x = np.delete(x, features_to_remove, axis=1)

        params['min_x'] = np.nanmin(x, axis=0)
        params['lambda_x'] = np.zeros(x.shape[1])
        params['mu_x'] = np.zeros(x.shape[1])
        params['sigma_x'] = np.zeros(x.shape[1])
        params['min_y'] = np.nanmin(y)
        params['lambda_y'] = np.zeros(y.shape[1])
        params['mu_y'] = np.zeros(y.shape[1])
        params['sigma_y'] = np.zeros(y.shape[1])

        x = x - params['min_x'][None, :] + 1

        for i in range(x.shape[1]):
            f = x[:, i]
            idx = np.isnan(f)
            f[~idx], params['lambda_x'][i] = boxcox(f[~idx])
            f[~idx] = zscore(f[~idx])
            x[:, i] = f

        y = y - params['min_y'] + np.finfo(float).eps

        for i in range(y.shape[1]):
            t = y[:, i]
            idx = np.isnan(t)
            t[~idx], params['lambda_y'][i] = boxcox(t[~idx])
            t[~idx] = zscore(t[~idx])
            y[:, i] = t

        new_features = pd.DataFrame(
            x,
            columns=features.drop(features_names, axis=1, inplace=False).columns,
            index=features.index
        )
        new_target = pd.DataFrame(
            y,
            columns=target.columns,
            index=target.index
        )

        return new_features, new_target
