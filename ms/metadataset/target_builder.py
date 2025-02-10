from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_handler import MetricsHandler
from ms.handler.metadata_source import MetadataSource, TabzillaSource
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

    @property
    def handler_path(self) -> str:
        return self.config.data_path

    @property
    def class_folder(self) -> str:
        return self.config.target_folder

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.metric_name = metric_name
        self.index_name = index_name
        self.alg_name = alg_name


    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        metric_results = self.__rearrange_dataset__(
            metrics_dataset=metrics_dataset
        )
        target_array = self.__get_target__(
            metrics_dataset=metric_results
        )
        target_cols = self.__get_col_names__(
            metrics_dataset=metric_results
        )
        handler_info = HandlerInfo(suffix=self.__get_suffix__())
        return (pd.DataFrame(target_array, columns=target_cols, index=metric_results.index),
                handler_info)

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

    @abstractmethod
    def __get_suffix__(self) -> str:
        pass


class TargetRawBuilder(TargetBuilder):
    @property
    def class_name(self) -> str:
        return "raw"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )

    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        target_raw = metrics_dataset.to_numpy()
        return target_raw

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        return metrics_dataset.columns

    def __get_suffix__(self) -> str:
        return self.class_name


class TargetPerfBuilder(TargetBuilder):
    @property
    def class_name(self) -> str:
        return "perf"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            perf_type: str = "abs", # or "rel"
            n_bins: int = 2,
            strategy: str = "quantile",
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )
        self.perf_type = perf_type
        self.n_bins = n_bins
        self.strategy = strategy

    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        target_perf = metrics_dataset.to_numpy(copy=True)
        target_perf = np.where(np.isnan(target_perf), -np.inf, target_perf)

        if self.perf_type == "abs":
            target_perf = self.__get_abs_perf__(nd_array=target_perf)
        elif self.perf_type == "rel":
            target_perf = self.__get_rel_perf__(nd_array=target_perf)
        else:
            raise ValueError(f"Unsupported performance metric: {self.perf_type}")

        return target_perf

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        cols = []
        for alg_name in metrics_dataset.columns:
            cols.append(f"{alg_name}__{self.perf_type}perf")
        return cols

    def __get_suffix__(self) -> str:
        return f"{self.class_name}_{self.perf_type}"

    def __get_abs_perf__(self, nd_array: NDArrayFloatT) -> NDArrayFloatT:
        new_array = np.zeros_like(nd_array)
        disc = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.strategy,
        )
        for i in range(nd_array.shape[1]):
            new_array[:, i] = disc.fit_transform(nd_array[:, i].reshape(-1, 1)).flatten()
        return new_array

    def __get_rel_perf__(self, nd_array: NDArrayFloatT) -> NDArrayFloatT:
        new_array = np.zeros_like(nd_array)
        for i in range(nd_array.shape[0]):
            row = np.argsort(nd_array[i])[::-1]
            for j in range(nd_array.shape[1]):
                new_array[i, row[j]] = j + 1
        disc = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.strategy,
        )
        new_array = disc.fit_transform(new_array.T).T
        return new_array


class TargetDiffBuilder(TargetBuilder):
    @property
    def class_name(self) -> str:
        return "diff"

    def __init__(
            self,
            md_source: MetadataSource,
            classes: list[str],
            model_classes: dict[str, str],
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            n_bins: int = 3,
            strategy: str = "quantile",
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )
        self.classes = classes
        self.model_classes = model_classes
        self.n_bins = n_bins
        self.strategy = strategy
        self._col_name = ""

    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        mean_vals = metrics_dataset.mean()
        max_res = {c : ("", 0.) for c in self.classes}
        for i in mean_vals.index:
            if mean_vals[i] > max_res[self.model_classes[i]][1]:
                max_res[self.model_classes[i]] = (i, mean_vals[i])
        models = [max_res[key][0] for key in max_res]

        diff_df = pd.DataFrame(index=metrics_dataset.index)
        diff_df[f"diff__{models[0]}__{models[1]}"] \
            = metrics_dataset[models[0]] - metrics_dataset[models[1]]

        disc = KBinsDiscretizer(
            n_bins=3,
            encode="ordinal",
            strategy="quantile",
        )
        diff_df.iloc[:, 0] = disc.fit_transform(X=diff_df)
        self._col_name = diff_df.columns[0]

        return diff_df.to_numpy(copy=True)

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        return [self._col_name]

    def __get_suffix__(self) -> str:
        return self.class_name


if __name__ == "__main__":
    raw_builder = TargetRawBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        test_mode=False,
    )

    abs_perf_builder = TargetPerfBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        perf_type="abs",
        n_bins=2,
        strategy="quantile",
        test_mode=False,
    )

    rel_perf_builder = TargetPerfBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        perf_type="rel",
        n_bins=3,
        strategy="uniform",
        test_mode=False,
    )

    model_classes = {
        "rtdl_FTTransformer": "nn",
        "rtdl_MLP": "nn",
        "rtdl_ResNet": "nn",
        "LinearModel": "classic",
        "RandomForest": "classic",
        "XGBoost": "classic"
    }

    diff_builder = TargetDiffBuilder(
        classes=["nn", "classic"],
        model_classes=model_classes,
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        n_bins=3,
        strategy="uniform",
        test_mode=False,
    )

    raw = raw_builder.handle_metrics()
    abs = abs_perf_builder.handle_metrics()
    rel = rel_perf = rel_perf_builder.handle_metrics()
    diff = diff_builder.handle_metrics()
