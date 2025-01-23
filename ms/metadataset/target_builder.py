from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ms.handler.metadata_handler import MetricsHandler
from ms.handler.metadata_source import MetadataSource
from ms.utils.calc_utils import AbsRangeStorage
from ms.utils.typing import NDArrayFloatT


class TargetBuilder(MetricsHandler, ABC):
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
            metrics_folder: str | None = "filtered",
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
        )
        self._md_source = md_source
        self.metric_name = metric_name
        self.index_name = index_name
        self.alg_name = alg_name


    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        metric_results = self.__rearrange_dataset__(
            metrics_dataset=metrics_dataset
        )
        target_array = self.__get_target__(
            metrics_dataset=metric_results
        )
        target_cols = self.__get_col_names__(
            metrics_dataset=metric_results
        )
        return pd.DataFrame(target_array, columns=target_cols, index=metric_results.index)

    def __rearrange_dataset__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        return metrics_dataset.pivot_table(
            values=self.metric_name,
            index=self.index_name,
            columns=self.alg_name,
            aggfunc='first'
        )

    @abstractmethod
    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        pass


class TargetRawBuilder(TargetBuilder):
    @property
    def class_name(self) -> str:
        return "target_raw_builder"

    @property
    def class_folder(self) -> str:
        return self.config.target_raw_folder

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )

    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        target_raw = metrics_dataset.to_numpy()
        return target_raw

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        return metrics_dataset.columns


class TargetPerfBuilder(TargetBuilder):
    @property
    def class_name(self) -> str:
        return "target_perf_builder"

    @property
    def class_folder(self) -> str:
        return self.config.target_perf_folder

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            perf_type: str = "abs", # or "rel"
            abs_ranges: AbsRangeStorage | None = None,
            rel_threshold: float = 0.05,
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )
        self.perf_type = perf_type
        self.abs_ranges = abs_ranges if abs_ranges is not None \
            else AbsRangeStorage()
        self.rel_threshold = rel_threshold


    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        target_perf = metrics_dataset.to_numpy()
        target_perf = np.where(np.isnan(target_perf), -np.inf, target_perf)

        if self.perf_type == "abs":
            self.__get_abs_perf__(nd_array=target_perf)
        elif self.perf_type == "rel":
            self.__get_rel_perf__(nd_array=target_perf)
        else:
            raise ValueError(f"Unsupported performance metric: {self.perf_type}")

        return target_perf

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        cols = []
        for alg_name in metrics_dataset.columns:
            cols.append(f"{alg_name}__{self.perf_type}perf")
        return cols

    def __get_abs_perf__(self, nd_array: NDArrayFloatT) -> None:
        for i in range(nd_array.shape[0]):
            for j in range(nd_array.shape[1]):
                for key in self.abs_ranges.storage:
                    right_value, left_value = self.abs_ranges.storage[key]
                    if right_value <= nd_array[i, j] < left_value:
                        nd_array[i, j] = float(key)
                        break

    def __get_rel_perf__(self, nd_array: NDArrayFloatT) -> None:
        best_algo = np.max(nd_array, axis=1)
        best_algo[best_algo == 0] = np.finfo(float).eps
        nd_array = (1 - (nd_array / best_algo[:, None]))
        for i in range(nd_array.shape[0]):
            for j in range(nd_array.shape[1]):
                if nd_array[i, j] <= self.rel_threshold:
                    nd_array[i, j] = 1.0
                else:
                    nd_array[i, j] = 0.0
