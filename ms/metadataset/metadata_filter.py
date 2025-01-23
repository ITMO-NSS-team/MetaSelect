from abc import ABC

import numpy as np
import pandas as pd

from ms.metadataset.metadata_handler import FeaturesHandler, MetricsHandler
from ms.metadataset.metadata_source import TabzillaSource


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
    def class_folder(self) -> str:
        return self.config.filtered_folder

    def __init__(
            self,
            features_folder: str = "formatted",
            metrics_folder: str | None = "formatted",
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
        )


class TabzillaFilter(MetadataFilter):
    @property
    def source(self) -> TabzillaSource:
        return TabzillaSource()

    def __init__(
            self,
            features_folder: str = "formatted",
            metrics_folder: str | None = "formatted",
            nan_threshold: float = 0.5,
            fill_func: str = "median",
            features_to_exclude: list[str] | None = None,
            keys_to_exclude: list[str] | None = None,
            datasets_to_exclude: list[str] | None = None,
            models_list: list[str] | None = None,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
        )
        self.nan_threshold = nan_threshold
        self.fill_func = fill_func
        self.features_to_exclude = features_to_exclude
        self.keys_to_exclude = keys_to_exclude
        self.datasets_to_exclude = datasets_to_exclude
        self.models_list = models_list


    def __handle_features__(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        filtered_features = features_dataset.copy()

        self.__remove_features_by_name__(features_dataset=filtered_features)
        self.__remove_datasets_by_name__(dataset=filtered_features)
        self.__remove_features_by_key__(features_dataset=filtered_features)
        self.__remove_unsuitable_features__(features_dataset=filtered_features)
        self.__fill_undefined_values__(features_dataset=filtered_features)

        return filtered_features

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        filtered_metrics = metrics_dataset.copy()

        self.__remove_datasets_by_name__(dataset=filtered_metrics)
        self.__filter_models__(metrics_dataset=filtered_metrics)
        self.__filter_datasets__(metrics_dataset=filtered_metrics)

        return filtered_metrics

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

    def __remove_features_by_name__(self, features_dataset: pd.DataFrame) -> None:
        if self.features_to_exclude is not None:
            features_dataset.drop(self.features_to_exclude, axis="columns", inplace=True)

    def __remove_datasets_by_name__(self, dataset: pd.DataFrame) -> None:
        if self.datasets_to_exclude is not None:
            dataset.drop(self.datasets_to_exclude, axis="index", inplace=True)

    def __remove_features_by_key__(self, features_dataset: pd.DataFrame) -> None:
        if self.keys_to_exclude is not None:
            features_to_remove = []
            for f in features_dataset.columns:
                for key in self.keys_to_exclude:
                    if key in f:
                        features_to_remove.append(f)
                        break
            features_dataset.drop(features_to_remove, axis="columns", inplace=True)

    def __filter_models__(self, metrics_dataset: pd.DataFrame) -> None:
        if self.models_list is not None:
            for index, row in metrics_dataset.iterrows():
                if row["alg_name"] not in self.models_list:
                    metrics_dataset.drop(index, axis="index", inplace=True)

    def __filter_datasets__(self, metrics_dataset: pd.DataFrame):
        if self.models_list is not None:
            dataset_models = {}
            for index, row in metrics_dataset.iterrows():
                if row["dataset_name"] not in dataset_models:
                    dataset_models[row["dataset_name"]] = set()
                dataset_models[row["dataset_name"]].add(row["alg_name"])
            for index, row in metrics_dataset.iterrows():
                if dataset_models[row["dataset_name"]] != set(self.models_list):
                    metrics_dataset.drop(index, axis="index", inplace=True)
