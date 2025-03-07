from pathlib import Path
from typing import Callable

import pandas as pd

from ms.handler.data_handler import DataHandler
from ms.handler.data_source import DataSource
from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selector_data import SelectorData
from ms.metaresearch.selectors.model_wrapper import RFESelector
from ms.utils.navigation import pjoin, load, save, rewrite_decorator


class MetaLearner(DataHandler):
    @property
    def class_suffix(self) -> str | None:
        return None

    @property
    def class_name(self) -> str:
        return "meta_learner"

    @property
    def class_folder(self) -> str:
        return self.config.meta_learning

    @property
    def source(self) -> DataSource:
        return self._md_source

    @property
    def has_index(self) -> dict:
        return {
            "features": True,
            "metrics": True,
        }

    @property
    def load_root(self) -> str:
        return self.config.resources

    @property
    def save_root(self) -> str:
        return self.config.results_path

    def __init__(
            self,
            md_source: DataSource,
            opt_scoring: str,
            model_scoring: dict[str, Callable],
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            opt_method: str | None = None,
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
        self.opt_method = opt_method
        self.opt_cv = opt_cv
        self.model_cv = model_cv
        self.n_trials = n_trials

    def run_models(
            self,
            models: list[MetaModel],
            feature_suffixes: list[str],
            target_suffixes: list[str],
            selector_names: list[str],
            to_rewrite: bool = True,
    ) -> None:
        selectors = self.load_selectors(
            features_suffixes=feature_suffixes,
            metrics_suffixes=target_suffixes,
            selector_names=selector_names,
        )
        for feature_suffix in feature_suffixes:
            print(f"Feature suffix: {feature_suffix}")
            x_df = self.load_features(suffix=feature_suffix)
            splits = self.load_samples(
                sample_type=self.config.splits_prefix,
                folders=[feature_suffix]
            )
            for s_name in selector_names:
                for target_suffix in target_suffixes:
                    selector = selectors[s_name][feature_suffix][target_suffix]
                    if selector.features_suffix != feature_suffix:
                        continue
                    print(f"Selector: {selector.name}")
                    print(f"Target file: metrics__{target_suffix}.csv")
                    y_df = self.load_metrics(suffix=target_suffix)
                    for model in models:
                        print(f"Metamodel: {model.name}")
                        if selector.features is None:
                            if model.name == "knn":
                                continue
                            rfe_handler = RFESelector(md_source=self.source, model=model)
                            selector = rfe_handler.perform(
                                features_suffix=feature_suffix,
                                metrics_suffix=target_suffix,
                                to_rewrite=to_rewrite,
                            )
                        for sample in selector.features:
                            print(f"Sample size: {sample}")
                            save_path = self.get_path(
                                root_type="save",
                                inner_folders=[
                                    feature_suffix,
                                    selector.name,
                                    target_suffix,
                                    model.name,
                                ],
                                prefix=self.get_file_name(
                                    prefix=self.config.results_prefix,
                                    suffix=sample,
                                ),
                                file_type="csv",
                            )
                            self.run_samples(
                                save_path=save_path,
                                x_df=x_df,
                                y_df=y_df,
                                selector=selector,
                                sample=sample,
                                splits=splits,
                                model=model,
                                to_rewrite=to_rewrite,
                            )

    @rewrite_decorator
    def run_samples(
            self,
            save_path: Path,
            x_df: pd.DataFrame,
            y_df: pd.DataFrame,
            selector: SelectorData,
            sample: dict,
            splits: dict,
            model: MetaModel,
    ):
        sample_res = []
        sample_params = {}
        for n_iter in selector.features[sample]:
            print(f"Iter: {n_iter}")
            model_scores = model.run(
                x=x_df,
                y=y_df,
                splits=splits,
                slices=selector.features[sample][n_iter],
                opt_scoring=self.opt_scoring,
                model_scoring=self.model_scoring,
                opt_method=self.opt_method,
                opt_cv=self.opt_cv,
                n_trials=self.n_trials,
            )

            formatted_scores, formatted_params = self.format_scores(
                model_scores=model_scores,
                n_samples=sample
            )
            sample_res.append(formatted_scores)
            sample_params[n_iter] = formatted_params
        sample_res = pd.concat(sample_res)

        save(
            data=sample_res,
            path=save_path,
            file_type="csv",
        )
        save(
            data=sample_params,
            path=Path(pjoin(
                save_path.parent,
                f"{model.name}.json"
            )),
            file_type="json",
        )


    def format_scores(
            self,
            model_scores: dict[str, dict],
            n_samples: int
    ) -> tuple[pd.DataFrame, dict]:
        res_df = pd.DataFrame()
        for model in model_scores.keys():
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
        res_df["samples"] = [n_samples for _ in range(len(res_df.index))]
        res_df.index.name = "model"

        best_params = {i:{} for i in model_scores.keys()}
        for model in model_scores.keys():
            best_params[model] = model_scores[model]["params"]

        return res_df, best_params

    @staticmethod
    def sjoin(x: pd.DataFrame) -> str:
        return ';'.join(x[x.notnull()].astype(str))

    def load_selectors(
            self,
            features_suffixes: list[str],
            metrics_suffixes: list[str],
            selector_names: list[str],
    ) -> list[SelectorData]:
        selectors = {}
        for features_suffix in features_suffixes:
            for metrics_suffix in metrics_suffixes:
                for s_name in selector_names:
                    if selectors.get(s_name) is None:
                        selectors[s_name] = {}
                    if selectors[s_name].get(features_suffix) is None:
                        selectors[s_name][features_suffix] = {}
                    json_path = self.get_path(
                        root_type="save",
                        inner_folders=[
                            features_suffix,
                            s_name,
                            self.config.selection_data
                        ],
                        prefix=metrics_suffix,
                        file_type="json"
                    )
                    if json_path.exists():
                        results = load(path=json_path, file_type="json")
                        selectors[s_name][features_suffix][metrics_suffix] = (
                            SelectorData(
                                name=s_name,
                                features_suffix=features_suffix,
                                metrics_suffix=metrics_suffix,
                                features=results
                            )
                        )
                    else:
                        selectors[s_name][features_suffix][metrics_suffix] = (
                            SelectorData(
                                name=s_name,
                                features_suffix=features_suffix,
                                metrics_suffix=metrics_suffix,
                                features=None
                            )
                        )
        return selectors
