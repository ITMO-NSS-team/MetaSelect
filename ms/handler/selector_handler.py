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
            rewrite: bool = False,
    ) -> SelectorData:
        slices = self.load_samples(
            file_name=f"{self.config.slices_prefix}",
            inner_folders=[features_suffix]
        )
        splits = self.load_samples(
            file_name=f"{self.config.splits_prefix}",
            inner_folders=[features_suffix]
        )

        features = self.load_features(suffix=features_suffix)
        metrics = self.load_metrics(suffix=metrics_suffix)
        target_models = [col for col in metrics.columns]
        results = {}
        errors = {}

        json_path = self.get_path(
            folder_name=features_suffix,
            file_name=f"{metrics_suffix}.json",
            inner_folders=[self.class_folder, "selection_data"]
        )
        if not rewrite and json_path.exists():
            return SelectorData(
                name = self.class_folder,
                features_suffix=features_suffix,
                metrics_suffix=metrics_suffix,
                features=self.load_json(
                    folder_name="",
                    file_name="",
                    path=json_path
                )
            )

        for f_slice in slices:
            print(f"Slice: {f_slice}")
            results[f_slice] = {}
            res_list = []
            res_path = self.get_path(
                folder_name=features_suffix,
                file_name=f"{f_slice}.csv",
                inner_folders=[self.class_folder, "selection_data", metrics_suffix]
            )
            if not rewrite and res_path.exists():
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
            self.save(
                data_frame=res_df,
                folder_name="",
                file_name="",
                path=res_path,
            )
        self.save_json(
            data=results,
            folder_name=features_suffix,
            file_name=f"{metrics_suffix}.json",
            inner_folders=[self.class_folder, "selection_data"]
        )
        self.save_json(
            data=errors,
            folder_name=features_suffix,
            file_name=f"{metrics_suffix}_errors.json",
            inner_folders=[self.class_folder, "selection_data"]
        )
        return SelectorData(
            name = self.class_folder,
            features_suffix=features_suffix,
            metrics_suffix=metrics_suffix,
            features=results
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
