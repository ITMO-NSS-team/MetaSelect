from abc import ABC

import numpy as np
import pandas as pd
from scipy.stats import boxcox, zscore, iqr

from ms.metadataset.metadata_handler import FeaturesHandler, MetricsHandler
from ms.metadataset.metadata_source import MetadataSource
from ms.utils.typing import NDArrayFloatT


class MetadataPreprocessor(FeaturesHandler, MetricsHandler, ABC):
    @property
    def class_name(self) -> str:
        return "preprocessor"

    @property
    def class_folder(self) -> str:
        return self.config.preprocessed_folder

    @property
    def source(self) -> MetadataSource:
        return self._md_source

    @property
    def has_index(self) -> dict:
        return {
            "features": True,
            "metrics": True,
        }

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "target_raw",
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
        )
        self._md_source = md_source


class PrelimPreprocessor(MetadataPreprocessor):
    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "raw",
            metrics_folder: str | None = "raw",
            perf_type: str = "abs",  # or "rel"
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
        )
        self.parameters = {}
        self.perf_type = perf_type

    def __handle_features__(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        x = features_dataset.to_numpy(copy=True)
        self.parameters['median'] = np.nanmedian(x, axis=0)
        self.parameters['iq_range'] = iqr(x, axis=0, nan_policy='omit')
        self.parameters['hi_bound'] = (self.parameters['median']
                                       + 5 * self.parameters['iq_range'])
        self.parameters['lo_bound'] = (self.parameters['median']
                                       - 5 * self.parameters['iq_range'])
        hi_mask = x > self.parameters['hi_bound'][None, :]
        lo_mask = x < self.parameters['lo_bound'][None, :]
        x = np.where(hi_mask, self.parameters['hi_bound'][None, :], x)
        x = np.where(lo_mask, self.parameters['lo_bound'][None, :], x)

        x, features_to_remove = self.__remove_constant_features__(
            nd_array=x,
            features_names=features_dataset.columns
        )

        self.parameters['min_x'] = np.nanmin(x, axis=0)
        self.parameters['lambda_x'] = np.zeros(x.shape[1])
        self.parameters['mu_x'] = np.zeros(x.shape[1])
        self.parameters['sigma_x'] = np.zeros(x.shape[1])

        x = x - self.parameters['min_x'][None, :] + 1

        for i in range(x.shape[1]):
            f = x[:, i]
            idx = np.isnan(f)
            f[~idx], self.parameters['lambda_x'][i] = boxcox(f[~idx])
            f[~idx] = zscore(f[~idx])
            x[:, i] = f

        new_features = pd.DataFrame(
            x,
            columns=features_dataset.drop(
                features_to_remove, axis=1, inplace=False
            ).columns,
            index=features_dataset.index
        )

        return new_features

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        y = metrics_dataset.copy().to_numpy()
        if self.perf_type == "rel":
            best_algo = np.max(np.where(np.isnan(y), -np.inf, y), axis=1)
            y[y == 0] = np.finfo(float).eps
            y = 1 - (y / best_algo[:, None])

        self.parameters['min_y'] = np.nanmin(y)
        self.parameters['lambda_y'] = np.zeros(y.shape[1])
        self.parameters['mu_y'] = np.zeros(y.shape[1])
        self.parameters['sigma_y'] = np.zeros(y.shape[1])

        y = y - self.parameters['min_y'] + np.finfo(float).eps

        for i in range(y.shape[1]):
            t = y[:, i]
            idx = np.isnan(t)
            t[~idx], self.parameters['lambda_y'][i] = boxcox(t[~idx])
            t[~idx] = zscore(t[~idx])
            y[:, i] = t

        new_target = pd.DataFrame(
            y,
            columns=metrics_dataset.columns,
            index=metrics_dataset.index
        )

        return new_target

    @staticmethod
    def __remove_constant_features__(
            nd_array: NDArrayFloatT,
            features_names: list[str],
    ) -> tuple[NDArrayFloatT, list[str]]:
        indexes_to_remove = []
        features_to_remove = []
        for i in range(nd_array.shape[1]):
            x_i = nd_array[:, i]
            if np.all(x_i == x_i[0]):
                indexes_to_remove.append(i)
                features_to_remove.append(features_names[i])

        return (np.delete(nd_array, indexes_to_remove, axis=1),
                features_to_remove)
