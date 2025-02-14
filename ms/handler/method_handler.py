from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd

from ms.handler.metadata_handler import MetricsHandler, FeaturesHandler
from ms.handler.metadata_source import MetadataSource
from ms.utils.typing import NDArrayFloatT


class MethodHandler(FeaturesHandler, MetricsHandler, ABC):
    @property
    def source(self) -> MetadataSource:
        return self._md_source

    @property
    def has_index(self) -> dict[str, bool]:
        return {
            "features": True,
            "metrics": True
        }

    @property
    def save_path(self) -> str:
        return self.config.results_path

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "processed",
            metrics_folder: str | None = "processed",
            method_name: str = "base_name",
            test_mode: bool = False
    ) -> None:
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.method_name = method_name

    def perform(
            self,
            features_suffix: str,
            metrics_suffix: str,
            method_config: dict | None = None
    ) -> None:
        features = self.load_features(suffix=features_suffix)
        metrics = self.load_metrics(suffix=metrics_suffix)

        results, file_name = self.__perform__(
            features_dataset=features,
            metrics_dataset=metrics,
            method_config=method_config,
        )
        self.save(
            data_frame=results,
            folder_name=self.get_name(self.class_folder),
            file_name=file_name
        )

    def __perform__(
            self,
            features_dataset: pd.DataFrame,
            metrics_dataset: pd.DataFrame,
            method_config: dict | None = None,
    ) -> tuple[pd.DataFrame, str]:
        x = features_dataset.to_numpy(copy=True)
        y = metrics_dataset.to_numpy(copy=True)

        method_name = self.method_name if method_config is None \
            else method_config["method_name"]

        if method_config is None or method_config["out_type"] == "multi":
            out_type = "multi"
            res_df = self.__multioutput_runner__(
                method=self.methods[method_name],
                x=x,
                y=y,
                features_names=features_dataset.columns,
                models_names=metrics_dataset.columns,
                method_config=method_config,
            )
        else:
            out_type = "single"
            res_df = self.methods[method_name](
                x=x,
                y=y,
                features_names=features_dataset.columns,
                method_config=method_config,
            )
        res_df.index.name = "dataset_name"
        return res_df, f"{method_name}_{out_type}.csv"

    def __handle_features__(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        return features_dataset

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        return metrics_dataset

    @staticmethod
    def __multioutput_runner__(
            method: Callable,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            models_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        res_df = pd.DataFrame(index=features_names)
        for i, model_name in enumerate(models_names):
            model_df = method(
                x=x,
                y=y[:, i],
                features_names=features_names,
                method_config=method_config,
            )
            model_df.columns = [f"{i}_{model_name}" for i in model_df.columns]
            res_df = pd.concat([res_df, model_df], axis=1)
        res_df.dropna(axis="index", how="any", inplace=True)
        return res_df

    @property
    @abstractmethod
    def methods(self) -> dict[str, Callable]:
        ...
