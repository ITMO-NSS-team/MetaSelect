from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ms.handler.data_handler import DataHandler
from ms.handler.data_source import DataSource
from ms.metaresearch.selector_data import SelectorData
from ms.utils.navigation import rewrite_decorator, save, load
from ms.utils.typing import NDArrayFloatT


class SelectorHandler(DataHandler, ABC):
    @property
    def source(self) -> DataSource:
        return self._md_source

    @property
    def class_suffix(self) -> str | None:
        return None

    @property
    def load_root(self) -> str:
        return self.config.resources

    @property
    def save_root(self) -> str:
        return self.config.results_path

    def __init__(
            self,
            md_source: DataSource,
            features_folder: str = "processed",
            metrics_folder: str | None = "processed",
            out_type: str = "multi",
            test_mode: bool = False
    ) -> None:
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.out_type = out_type

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
            to_rewrite: bool = False,
    ) -> SelectorData:
        slices = self.load_samples(
            sample_type=self.config.slices_prefix,
            folders=[features_suffix],
        )
        splits = self.load_samples(
            sample_type=self.config.splits_prefix,
            folders=[features_suffix]
        )

        features = self.load_features(suffix=features_suffix)
        metrics = self.load_metrics(suffix=metrics_suffix)
        target_models = [col for col in metrics.columns]

        json_path = self.get_path(
            root_type="save",
            inner_folders=[features_suffix, self.class_folder, "selection_data"],
            prefix=metrics_suffix,
            file_type="json",
        )
        errors_path = self.get_path(
            root_type="save",
            inner_folders=[features_suffix, self.class_folder, "selection_data"],
            prefix=f"{metrics_suffix}_errors",
            file_type="json",
        )

        self.run_selectors(
            save_path=json_path,
            errors_path=errors_path,
            slices=slices,
            splits=splits,
            features=features,
            metrics=metrics,
            features_suffix=features_suffix,
            metrics_suffix=metrics_suffix,
            target_models=target_models,
            to_rewrite=to_rewrite,
        )

        return SelectorData(
            name=self.class_folder,
            features_suffix=features_suffix,
            metrics_suffix=metrics_suffix,
            features=load(path=json_path, file_type="json")
        )


    @rewrite_decorator
    def run_selectors(
            self,
            save_path: Path,
            errors_path: str,
            slices: dict,
            splits: dict,
            features: pd.DataFrame,
            metrics: pd.DataFrame,
            features_suffix: str,
            metrics_suffix: str,
            target_models: list[str],
            to_rewrite: bool = False,
    ) -> None:
        results = {}
        errors = {}
        for f_slice in slices:
            print(f"Slice: {f_slice}")
            results[f_slice] = {}
            res_list = []
            res_path = (
                self.get_path(
                    root_type="save",
                    inner_folders=[
                        features_suffix,
                        self.class_folder,
                        "selection_data",
                        metrics_suffix
                    ],
                    prefix=f_slice,
                    file_type="json",
                )
            )
            if not to_rewrite and res_path.exists():
                df = pd.read_csv(res_path, index_col=0)
                for n_iter in slices[f_slice]:
                    results[f_slice][n_iter] = {}
                    for fold in splits:
                        results[f_slice][n_iter][fold] = {}
                        for k in range(len(target_models)):
                            idx = int(n_iter) * len(target_models) + k
                            results[f_slice][n_iter][fold][target_models[k]] \
                                = df.iloc[:, idx].dropna(how="any").index.tolist()
                continue
            for n_iter in slices[f_slice]:
                print(f"Iteration: {n_iter}")
                results[f_slice][n_iter] = {}
                for fold in splits:
                    print(f"Fold: {fold}")
                    train = splits[fold]["train"]
                    df, file_name = self.__perform__(
                        features_dataset=features.iloc[
                            train,
                            (slices[f_slice][n_iter])
                        ],
                        metrics_dataset=metrics.iloc[train, :],
                    )
                    results[f_slice][n_iter][fold] = {}
                    for k, target_model in enumerate(target_models):
                        selected_features = df.iloc[:, k].dropna(how="any").index.tolist()
                        if len(selected_features) == 0:
                            errors[f"{f_slice}_{n_iter}_{fold}_{target_model}"] = 0
                        results[f_slice][n_iter][fold][target_model] = selected_features
                    df.columns = [f"{col}_{n_iter}" for col in df.columns]
                    res_list.append(df)
            res_df = pd.concat(res_list, axis=1)
            save(
                data=res_df,
                path=res_path,
                file_type="csv",
            )
        save(
            data=results,
            path=save_path,
            file_type="json",
        )
        save(
            data=errors,
            path=errors_path,
            file_type="json",
        )

    def __perform__(
            self,
            features_dataset: pd.DataFrame,
            metrics_dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, str]:
        x = features_dataset.to_numpy(copy=True)
        y = metrics_dataset.to_numpy(copy=True)

        if self.out_type == "multi":
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
        return res_df, f"{self.class_name}_{out_type}.csv"


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
