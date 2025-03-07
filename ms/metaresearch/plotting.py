import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ms.handler.data_handler import DataHandler
from ms.handler.data_source import DataSource
from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selector_data import SelectorData
from ms.utils.navigation import rewrite_decorator, load


class Plotter(DataHandler):
    @property
    def class_suffix(self) -> str | None:
        return None

    @property
    def class_name(self) -> str:
        return "plotter"

    @property
    def class_folder(self) -> str:
        return self.config.plots

    @property
    def source(self) -> DataSource:
        return self._md_source

    @property
    def save_root(self) -> str:
        return self.config.plots_path

    @property
    def load_root(self) -> str:
        return self.config.results_path

    def __init__(
            self,
            md_source: DataSource,
            mean_cols: list[str],
            std_cols: list[str],
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            test_mode: bool = False,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.mean_cols = mean_cols
        self.std_cols = std_cols

    def plot(
            self,
            models: list[MetaModel],
            feature_suffixes: list[str],
            target_suffixes: list[str],
            selectors: list[SelectorData],
            target_models: list[str],
            metric:str = "test_f1_mean",
            to_rewrite: bool = False,
    ):
        for feature_suffix in feature_suffixes:
            selector_res = {}
            for selector in selectors:
                target_res = {}
                for target_suffix in target_suffixes:
                    model_res = {}
                    for model in models:
                        if model.name == "knn" and selector.name == "rfe":
                            continue
                        res_list = []
                        for sample in selector.features:
                            load_path = self.get_path(
                                root_type="load",
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
                                file_type="csv"
                            )
                            res = load(
                                path=load_path,
                                file_type="csv"
                            )
                            res.index = target_models * len(selector.features[sample])
                            res.index.name = "model"
                            res_list.append(res)
                        model_res[model.name] = pd.concat(res_list)[[metric, "samples"]]
                    target_res[target_suffix] = model_res
                    sel_res_path = self.get_path(
                        root_type="save",
                        inner_folders=[
                            feature_suffix,
                            selector.name,
                        ],
                        prefix=target_suffix,
                        file_type="png",
                    )
                    self.plot_selector_results(
                        selector_name=selector.name,
                        metamodels_res=model_res,
                        metric=metric,
                        save_path=sel_res_path,
                        to_rewrite=to_rewrite
                    )
                selector_res[selector.name] = target_res
            for target_suffix in target_suffixes:
                metamodels = [i.name for i in models]
                for metamodel in metamodels:
                    sel_comp_path = self.get_path(
                        root_type="save",
                        inner_folders=[
                            feature_suffix,
                            "plots",
                            target_suffix
                        ],
                        prefix=metamodel,
                        file_type="png"
                    )
                    self.plot_selector_comparison(
                        save_path=sel_comp_path,
                        sel_res=selector_res,
                        target_suffix=target_suffix,
                        metamodel=metamodel,
                        target_models=target_models,
                        metric=metric,
                        to_rewrite=to_rewrite,
                    )

    @staticmethod
    @rewrite_decorator
    def plot_selector_results(
            selector_name: str,
            metamodels_res: dict,
            metric: str,
            save_path: str,
    ):
        print(f"Plotting {selector_name}")
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        cells = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        for i, metamodel in enumerate(metamodels_res):
            cells[i].set(
                title=metamodel,
                ylabel=f"{metric[5:]}",
            )
            metamodels_res[metamodel].rename(
                {"samples": "init feature numbers"}, axis=1, inplace=True
            )
            sns.barplot(
                data=metamodels_res[metamodel],
                x="init feature numbers",
                y=metric,
                hue="model",
                ax=cells[i]
            )
            cells[i].legend()
        fig.tight_layout(pad=1.0)
        fig.suptitle(selector_name)
        plt.subplots_adjust(top=0.95)

        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @rewrite_decorator
    def plot_selector_comparison(
            self,
            save_path,
            sel_res,
            target_suffix,
            metamodel,
            target_models,
            metric,
    ):
        print(f"Plotting selectors comparison for {metamodel}")
        if len(target_models) > 1:
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        for i, ax in enumerate(fig.axes):
            target_model = target_models[i]
            target_model_res = []
            ax.set_ylabel(f"{metric[5:]}")
            ax.set_title(target_models[i])
            for sel in sel_res:
                if sel_res[sel][target_suffix].get(metamodel) is None:
                    continue
                df = sel_res[sel][target_suffix][metamodel]
                df = df.loc[target_model].to_frame().T
                df["selector"] = [sel for _ in range(len(df.index))]
                target_model_res.append(df)
            target_df = pd.concat(target_model_res)
            target_df.rename({"samples": "init feature numbers"}, axis=1, inplace=True)
            sns.barplot(
                data=target_df,
                x="init feature numbers",
                y=metric,
                hue="selector",
                ax=ax
            )
            ax.legend()
        fig.tight_layout(pad=0.9)
        fig.suptitle(metamodel)
        plt.subplots_adjust(top=0.95)
        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
