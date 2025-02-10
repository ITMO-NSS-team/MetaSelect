from abc import ABC
from statistics import median

import numpy as np
import pandas as pd

from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_handler import FeaturesHandler, MetricsHandler
from ms.handler.metadata_source import TabzillaSource
from ms.utils.metadata_utils import remove_constant_features


class MetadataFilter(FeaturesHandler, MetricsHandler, ABC):
    @property
    def class_name(self) -> str:
        return "filter"

    @property
    def has_index(self) -> dict:
        return {
            "features": True,
            "metrics": False,
        }

    @property
    def handler_path(self) -> str:
        return self.config.data_path

    @property
    def class_folder(self) -> str:
        return self.config.filtered_folder

    def __init__(
            self,
            features_folder: str = "formatted",
            metrics_folder: str | None = "formatted",
            test_mode: bool = False,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )


class TabzillaFilter(MetadataFilter):
    @property
    def source(self) -> TabzillaSource:
        return TabzillaSource()

    def __init__(
            self,
            features_folder: str = "formatted",
            metrics_folder: str | None = "formatted",
            test_mode: bool = False,
            nan_threshold: float = 0.5,
            fill_func: str = "median",
            funcs_to_exclude: list[str] | None = None,
            keys_to_exclude: list[str] | None = None,
            datasets_to_exclude: list[str] | None = None,
            models_list: list[str] | None = None,
            value_threshold: float = 10e6,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.nan_threshold = nan_threshold
        self.fill_func = fill_func
        self.funcs_to_exclude = funcs_to_exclude
        self.keys_to_exclude = keys_to_exclude
        self.datasets_to_exclude = datasets_to_exclude
        self.models_list = models_list
        self.value_threshold = value_threshold


    def __handle_features__(self, features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        filtered_features = features_dataset.copy()

        self.__remove_features_by_func__(features_dataset=filtered_features)
        self.__remove_datasets_by_name__(dataset=filtered_features)
        self.__remove_features_by_key__(features_dataset=filtered_features)
        self.__remove_unsuitable_features__(features_dataset=filtered_features)
        self.__filter_outliers__(features_dataset=filtered_features)
        self.__fill_undefined_values__(features_dataset=filtered_features)
        remove_constant_features(features_dataset=filtered_features)

        return filtered_features, HandlerInfo()

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        filtered_metrics = metrics_dataset.copy()

        self.__remove_datasets_by_name__(dataset=filtered_metrics)
        self.__filter_models__(metrics_dataset=filtered_metrics)
        self.__filter_datasets_by_model__(metrics_dataset=filtered_metrics)

        return filtered_metrics, HandlerInfo()

    def __remove_unsuitable_features__(self, features_dataset: pd.DataFrame) -> None:
        num_datasets = len(features_dataset.index)
        for col in features_dataset:
            x = features_dataset[col].to_numpy()
            if (features_dataset[col].isna().sum() > num_datasets * self.nan_threshold
                    or np.all(x == x[0])):
                features_dataset.drop(col, axis="columns", inplace=True)

    def __fill_undefined_values__(self, features_dataset: pd.DataFrame) -> None:
        if self.fill_func == "median":
            values = features_dataset.median(numeric_only=True)
        else:
            values = features_dataset.mean(numeric_only=True)
        features_dataset.fillna(values, inplace=True)

    def __remove_features_by_func__(self, features_dataset: pd.DataFrame) -> None:
        if self.funcs_to_exclude is not None:
            features_to_remove = []
            for feature in features_dataset.columns:
                f_name = feature.split(".")
                if len(f_name) == 3:
                    continue
                f_func = f_name[3]
                if f_name[-1] == "relative":
                    features_to_remove.append(feature)
                for key in self.funcs_to_exclude:
                    if f_func == key:
                        features_to_remove.append(feature)
            features_dataset.drop(features_to_remove, axis="columns", inplace=True)

    def __remove_datasets_by_name__(self, dataset: pd.DataFrame) -> None:
        if self.datasets_to_exclude is not None:
            dataset.drop(self.datasets_to_exclude, axis="index", inplace=True)

    def __remove_features_by_key__(self, features_dataset: pd.DataFrame) -> None:
        if self.keys_to_exclude is not None:
            features_to_remove = []
            for feature in features_dataset.columns:
                for key in self.keys_to_exclude:
                    if key in feature:
                        features_to_remove.append(feature)
                        break
            features_dataset.drop(features_to_remove, axis="columns", inplace=True)

    def __filter_outliers__(self, features_dataset: pd.DataFrame) -> None:
        outliers_dict = {}
        outliers_list = []

        for i, feature in enumerate(features_dataset.columns):
            feature_outliers = []
            for j, val in enumerate(features_dataset[feature]):
                if val > self.value_threshold or val < -self.value_threshold:
                    feature_outliers.append(j)
            if len(feature_outliers) > 1:
                outliers_dict[feature] = feature_outliers
                outliers_list.append(len(feature_outliers))
            elif len(feature_outliers) == 1:
                outliers_dict[feature] = feature_outliers
            else:
                pass
        median_outlier_count = median(outliers_list)

        features_to_drop = []
        for feature in features_dataset.columns:
            if (outliers_dict.get(feature) is not None
                    and len(outliers_dict[feature]) > median_outlier_count):
                outliers_dict.pop(feature)
                features_to_drop.append(feature)
        features_dataset.drop(features_to_drop, axis="columns", inplace=True)

        datasets_to_drop = set()
        for feature in outliers_dict:
            for dataset_idx in outliers_dict[feature]:
                datasets_to_drop.add(dataset_idx)
        datasets_to_drop = [features_dataset.index[i] for i in datasets_to_drop]
        features_dataset.drop(datasets_to_drop, axis="index", inplace=True)

    def __filter_models__(self, metrics_dataset: pd.DataFrame) -> None:
        if self.models_list is not None:
            for index, row in metrics_dataset.iterrows():
                if row["alg_name"] not in self.models_list:
                    metrics_dataset.drop(index, axis="index", inplace=True)

    def __filter_datasets_by_model__(self, metrics_dataset: pd.DataFrame):
        if self.models_list is not None:
            dataset_models = {}
            for index, row in metrics_dataset.iterrows():
                if row["dataset_name"] not in dataset_models:
                    dataset_models[row["dataset_name"]] = set()
                dataset_models[row["dataset_name"]].add(row["alg_name"])
            for index, row in metrics_dataset.iterrows():
                if dataset_models[row["dataset_name"]] != set(self.models_list):
                    metrics_dataset.drop(index, axis="index", inplace=True)


if __name__ == "__main__":
    md_filter = TabzillaFilter(
        features_folder="formatted",
        metrics_folder="formatted",
        funcs_to_exclude=[
            "count",
            "histogram",
            "iq_range",
            "median",
            "quantiles",
            "range",
        ],
        models_list=["XGBoost", "RandomForest", "LinearModel",
                     "rtdl_ResNet", "rtdl_FTTransformer", "rtdl_MLP"],
        test_mode=False,
        value_threshold=1e6,
    )

    f = md_filter.handle_features(to_save=True)
    print(f.shape)
    m = md_filter.handle_metrics(to_save=True)
    print(m.shape)
