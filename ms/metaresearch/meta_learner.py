import json
import os.path
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ms.handler.metadata_handler import MetadataHandler
from ms.handler.metadata_source import MetadataSource
from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selector_data import SelectorData
from ms.utils.navigation import pjoin

class MetaLearner(MetadataHandler):
    @property
    def class_name(self) -> str:
        return "meta_learner"

    @property
    def class_folder(self) -> str:
        return self.config.meta_learning

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
    def save_path(self) -> str:
        return self.config.results_path

    def __init__(
            self,
            md_source: MetadataSource,
            opt_scoring: str,
            model_scoring: dict[str, Callable],
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            use_optuna: bool = True,
            opt_cv: int = 5,
            model_cv: int = 10,
            n_trials: int = 50,
            test_mode: bool = False,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.opt_scoring = model_scoring[opt_scoring]
        self.model_scoring = model_scoring
        self.opt_method = "optuna" if use_optuna else "grid_search"
        self.opt_cv = opt_cv
        self.model_cv = model_cv
        self.n_trials = n_trials

    def run_models(
            self,
            models: list[MetaModel],
            feature_suffix: str,
            target_suffixes: list[str],
            selectors: list[SelectorData],
            rewrite: bool = True,
    ):
        x_df = self.load_features(suffix=feature_suffix)
        for selector in selectors:
            print(f"Selector: {selector.name}")
            x_selected = selector.get_features(x=x_df)
            for target_suffix in target_suffixes:
                print(f"Target file: metrics__{target_suffix}.csv")
                y_df = self.load_metrics(suffix=target_suffix)
                for model in models:
                    print(f"Metamodel: {model.name}")
                    save_path = self.get_save_path(
                        folder_name=selector.name,
                        inner_folders=[target_suffix],
                        file_name=self.get_file_name(
                            prefix=self.config.results_prefix,
                            suffix=model.name
                        ),
                    )
                    if os.path.isfile(save_path) and not rewrite:
                        continue

                    model_scores = model.run(
                        x=x_selected,
                        y=y_df,
                        opt_scoring=self.opt_scoring,
                        model_scoring=self.model_scoring,
                        opt_method=self.opt_method,
                        opt_cv=self.opt_cv,
                        model_cv=self.model_cv,
                        n_trials=self.n_trials,
                    )

                    formatted_scores, formatted_params = self.format_scores(
                        model_scores=model_scores
                    )

                    self.save(
                        data_frame=formatted_scores,
                        folder_name=selector.name,
                        inner_folders=[target_suffix],
                        file_name=self.get_file_name(
                            prefix=self.config.results_prefix,
                            suffix=model.name
                        ),
                    )
                    self.save_params(
                        params=formatted_params,
                        save_path=save_path,
                        model_name=model.name
                    )


    def format_scores(self, model_scores: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
        res_df = pd.DataFrame()
        for model in model_scores.keys():
            model_scores[model]["cv"]["fit_score_time"] = list(
                np.array(model_scores[model]["cv"]["fit_time"]) +
                np.array(model_scores[model]["cv"]["score_time"])
            )
            model_scores[model]["cv"].pop("fit_time")
            model_scores[model]["cv"].pop("score_time")
            cur_df_mean = pd.DataFrame(model_scores[model]["cv"])
            new_cols_mean = [f"{i}_mean" for i in cur_df_mean.columns]
            cur_df_mean.columns = new_cols_mean

            cur_df_std = pd.DataFrame(model_scores[model]["cv"])
            new_cols_std = [f"{i}_std" for i in cur_df_std.columns]
            cur_df_std.columns = new_cols_std

            res_df = pd.concat([
                res_df,
                cur_df_mean.mean().to_frame(),
                cur_df_std.std().to_frame()
            ], axis=1)
            res_df.rename(columns={0: model}, inplace=True)
        res_df = res_df.groupby(level=0, axis=1).apply(lambda x: x.apply(self.sjoin, axis=1)).T
        res_df.index.name = "model"

        best_params = {i:{} for i in model_scores.keys()}
        for model in model_scores.keys():
            best_params[model] = model_scores[model]["params"]

        return res_df, best_params

    @staticmethod
    def save_params(params: dict, save_path: Path, model_name: str) -> None:
        with open(pjoin(save_path.parent, f"{model_name}.json"), "w") as f:
            json.dump(params, f)

    @staticmethod
    def sjoin(x: pd.DataFrame) -> str:
        return ';'.join(x[x.notnull()].astype(str))
