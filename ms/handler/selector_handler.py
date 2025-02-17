from abc import ABC, abstractmethod

import pandas as pd

from ms.handler.metadata_handler import MetadataHandler
from ms.handler.metadata_source import MetadataSource
from ms.metaresearch.selector_data import SelectorData
from ms.utils.typing import NDArrayFloatT


class SelectorHandler(MetadataHandler, ABC):
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
            test_mode: bool = False
    ) -> None:
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source

    @abstractmethod
    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        ...

    def perform(
            self,
            features_suffix: str,
            metrics_suffix: str,
            selector_config: dict | None = None,
    ) -> SelectorData:
        features = self.load_features(suffix=features_suffix)
        metrics = self.load_metrics(suffix=metrics_suffix)
        samples = self.load_samples(suffix=features_suffix)
        results = {}
        selector_name = selector_config["method_name"]

        for sample in samples:
            results[sample] = {}
            for n_iter in samples[sample]:
                df, file_name = self.__perform__(
                    features_dataset=features.loc[:, samples[sample][n_iter]],
                    metrics_dataset=metrics,
                    selector_name=selector_name,
                    selector_config=selector_config,
                )
                results[sample][n_iter] = list(df.index)
        self.save_json(
            data=results,
            folder_name=features_suffix,
            file_name=f"{metrics_suffix}.json",
            inner_folders=[selector_name, "selection_data"]
        )
        return SelectorData(
            name = selector_name,
            features_suffix=features_suffix,
            metrics_suffix=metrics_suffix,
            features=results
        )

    def __perform__(
            self,
            features_dataset: pd.DataFrame,
            metrics_dataset: pd.DataFrame,
            selector_name: str,
            selector_config: dict | None = None,
    ) -> tuple[pd.DataFrame, str]:
        x = features_dataset.to_numpy(copy=True)
        y = metrics_dataset.to_numpy(copy=True)

        method_name = selector_name if selector_config is None \
            else selector_config["method_name"]

        if selector_config is None or selector_config["out_type"] == "multi":
            out_type = "multi"
            res_df = self.__multioutput_runner__(
                x=x,
                y=y,
                features_names=features_dataset.columns,
                models_names=metrics_dataset.columns,
            )
        else:
            out_type = "single"
            res_df = self.handle_data(
                x=x,
                y=y,
                features_names=features_dataset.columns,
            )
        res_df.index.name = "dataset_name"
        return res_df, f"{method_name}_{out_type}.csv"


    def __multioutput_runner__(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            models_names: list[str],
    ) -> pd.DataFrame:
        res_df = pd.DataFrame(index=features_names)
        for i, model_name in enumerate(models_names):
            model_df = self.handle_data(
                x=x,
                y=y[:, i],
                features_names=features_names,
            )
            model_df.columns = [f"{i}_{model_name}" for i in model_df.columns]
            res_df = pd.concat([res_df, model_df], axis=1)
        res_df.dropna(axis="index", how="all", inplace=True)
        return res_df
