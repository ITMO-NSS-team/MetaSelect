from abc import ABC

import pandas as pd

from ms.handler.metadata_handler import FeaturesHandler, MetricsHandler
from ms.handler.metadata_source import TabzillaSource


class MetadataFormatter(FeaturesHandler, MetricsHandler, ABC):
    @property
    def class_name(self) -> str:
        return "formatter"

    @property
    def class_folder(self) -> str:
        return self.config.formatted_folder


    def __init__(
            self,
            features_folder: str = "raw",
            metrics_folder: str | None = "raw",
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
        )


class TabzillaFormatter(MetadataFormatter):
    @property
    def source(self) -> TabzillaSource:
        return TabzillaSource()

    @property
    def has_index(self) -> dict:
        return {
            "features": False,
            "metrics": False,
        }

    def __init__(
            self,
            features_folder: str = "raw",
            metrics_folder: str | None = "raw",
            agg_func_features: str = "median",
            agg_func_metrics: str = "mean",
            round_attrs: list[str] | None = None,
            filter_families: list[str] | None = None,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
        )
        self.agg_func_features = agg_func_features
        self.agg_func_metrics = agg_func_metrics
        self.round_attrs = round_attrs if round_attrs is not None else \
            [
                "f__pymfe.general.nr_inst",
                "f__pymfe.general.nr_attr",
                "f__pymfe.general.nr_bin",
                "f__pymfe.general.nr_cat",
                "f__pymfe.general.nr_num",
                "f__pymfe.general.nr_class",
            ]
        self.filter_families = filter_families

    def __handle_features__(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        agg_features = self.__aggregate_features__(features_dataset=features_dataset)
        self.__round_attributes__(features_dataset=agg_features)
        self.__filter_families__(features_dataset=agg_features)
        return agg_features

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        agg_metrics = self.__aggregate_metrics__(metrics_dataset=metrics_dataset)
        return agg_metrics

    def __aggregate_features__(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        agg_features = features_dataset.groupby("dataset_name")
        if self.agg_func_features == "median":
            agg_features = agg_features.median(numeric_only=True)
        else:
            agg_features = agg_features.mean(numeric_only=True)
        return agg_features

    def __round_attributes__(self, features_dataset: pd.DataFrame) -> None:
        if self.round_attrs is None:
            return
        for attr in self.round_attrs:
            if attr in features_dataset.columns:
                features_dataset.loc[:, attr] = features_dataset[attr].round(0)

    def __filter_families__(self, features_dataset: pd.DataFrame) -> None:
        if self.filter_families is None:
            return
        prefixes = [f"f__pymfe.{family}" for family in self.filter_families]
        filter_cols = [
            col
            for col in features_dataset.columns
            if not col.startswith("f__")
                or any(col.startswith(prefix) for prefix in prefixes)
        ]
        features_dataset.drop(columns=filter_cols, inplace=True)

    def __aggregate_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        agg_metrics = metrics_dataset.loc[
            metrics_dataset["hparam_source"] == "default"
        ].groupby(
            ["dataset_name", "alg_name"]
        )
        if self.agg_func_metrics == "median":
            agg_metrics = agg_metrics.median(numeric_only=True)
        else:
            agg_metrics = agg_metrics.mean(numeric_only=True)
        return agg_metrics
