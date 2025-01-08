from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TargetType(ABC):
    name: str = "target"
    def __init__(
            self,
            index_name: str = "dataset_name",
            alg_name: str = "alg_name"
    ) -> None:
        self.index_name = index_name
        self.alg_name = alg_name

    def get_target(
            self,
            model_results: pd.DataFrame,
            model_names: list[str],
            metric_name: str
    ) -> pd.DataFrame:
        metric_results = self.__get_metric__(
            model_results=model_results,
            metric_name=metric_name
        )
        target_array = self.__get_target__(
            model_results=metric_results,
            model_names=model_names
        )
        target_cols = self.__get_col_names__(
            model_names=model_names
        )
        return pd.DataFrame(target_array, columns=target_cols, index=metric_results.index)

    def __get_metric__(
            self,
            model_results: pd.DataFrame,
            metric_name: str
    ) -> pd.DataFrame:
        return model_results.pivot_table(
            values=metric_name,
            index=self.index_name,
            columns=self.alg_name,
            aggfunc='first'
        )

    @abstractmethod
    def __get_target__(
            self,
            model_results: pd.DataFrame,
            model_names: list[str]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def __get_col_names__(
            self,
            model_names: list[str]
    ) -> list[str]:
        pass


class TargetDiff(TargetType):
    name: str = "diff"
    def __init__(
            self,
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            target_type: str = "delta", # or "label"
            to_abs: bool = False,
            atol: float = 1e-3,
    ) -> None:
        super().__init__(index_name=index_name, alg_name=alg_name)
        self.target_type = target_type
        self.to_abs = to_abs
        self.atol = atol


    def __get_target__(
            self,
            model_results: pd.DataFrame,
            model_names: list[str]
    ) -> np.ndarray:
        model_a = model_results[model_names[0]].to_numpy()
        model_b = model_results[model_names[1]].to_numpy()
        if self.target_type == "delta":
            target = self.get_delta(a=model_a, b=model_b)
        elif self.target_type == "label":
            target = self.get_label(a=model_a, b=model_b)
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")

        return target

    def __get_col_names__(
            self,
            model_names: list[str]
    ) -> list[str]:
        return [
            f"{self.name}__{model_names[0]}__{model_names[1]}",
        ]

    def get_delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a - b) if self.to_abs else a - b

    def get_label(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        labels = a - b
        for i in range(labels.shape[0]):
            if np.isclose(labels[i], 0, atol=self.atol):
                labels[i] = 0
            elif labels[i] > 0:
                labels[i] = 1
            else:
                labels[i] = 2
        return labels


class TargetPerf(TargetType):
    name: str = "perf"
    def __init__(
            self,
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            perf_type: str = "abs", # or "rel"
            abs_threshold: float = 0.8,
            rel_threshold: float = 0.05,
    ):
        super().__init__(index_name=index_name, alg_name=alg_name)
        self.perf_type = perf_type
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold

    def __get_target__(
            self,
            model_results: pd.DataFrame,
            model_names: list[str]
    ) -> np.ndarray:
        target_raw = model_results.to_numpy()
        target_res = np.where(np.isnan(target_raw), -np.inf, target_raw)
        best_algo = np.max(target_res, axis=1)
        if self.perf_type == "abs":
            target_res = target_res >= self.abs_threshold
        elif self.perf_type == "rel":
            best_algo[best_algo == 0] = np.finfo(float).eps
            target_res = (1 - (target_res / best_algo[:, None])) <= self.rel_threshold
        else:
            raise ValueError(f"Unsupported performance metric: {self.perf_type}")

        return target_res


    def __get_col_names__(
            self,
            model_names: list[str]
    ) -> list[str]:
        col_names = []
        for model_name in model_names:
            col_names.append(
                f"{self.perf_type}{self.name}__{model_name}"
            )

        return col_names
