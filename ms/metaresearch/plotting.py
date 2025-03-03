import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ms.handler.data_handler import DataHandler
from ms.handler.data_source import DataSource
from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selector_data import SelectorData


class Plotter(DataHandler):
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
    def has_index(self) -> dict:
        return {
            "features": True,
            "metrics": True,
        }

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
            rewrite: bool = False,
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
        self.rewrite = rewrite

    def plot(
            self,
            models: list[MetaModel],
            feature_suffixes: list[str],
            target_suffixes: list[str],
            selectors: list[SelectorData],
            target_models: list[str],
            metric:str = "test_f1_mean",
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
                                folder_name=feature_suffix,
                                inner_folders=[
                                    selector.name,
                                    target_suffix,
                                    model.name,
                                ],
                                file_name=self.get_file_name(
                                    prefix=self.config.results_prefix,
                                    suffix=sample,
                                ),
                                to_save=False
                            )
                            res = self.load(
                                folder_name="",
                                file_name="",
                                path=load_path,
                            )
                            res.index = target_models * len(selector.features[sample])
                            res.index.name = "model"
                            res_list.append(res)
                        model_res[model.name] = pd.concat(res_list)[[metric, "samples"]]
                    target_res[target_suffix] = model_res
                    sel_res_path = self.get_path(
                        folder_name=feature_suffix,
                        inner_folders=[
                            selector.name,
                            # target_suffix,
                        ],
                        file_name=f"{target_suffix}.png",
                    )
                    if sel_res_path.exists() and not self.rewrite:
                        continue
                    os.makedirs(os.path.dirname(sel_res_path), exist_ok=True)
                    self.plot_selector_results(
                        selector_name=selector.name,
                        metamodels_res=model_res,
                        metric=metric,
                        save_path=sel_res_path
                    )
                selector_res[selector.name] = target_res
            for target_suffix in target_suffixes:
                self.plot_selector_comparison(
                    selector_res,
                    feature_suffix,
                    target_suffix,
                    [i.name for i in models],
                    target_models,
                    metric,
                )

    @staticmethod
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
            metamodels_res[metamodel].rename({"samples": "init feature numbers"}, axis=1, inplace=True)
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

    def plot_selector_comparison(
            self,
            sel_res,
            feature_suffix,
            target_suffix,
            metamodels,
            target_models,
            metric,
    ):
        for metamodel in metamodels:
            print(f"Plotting selectors comparison for {metamodel}")
            if len(target_models) > 1:
                fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            else:
                fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            for i, ax in enumerate(fig.axes):
                # target_model_res = {
                #     i: {} for i in sel_res
                # }
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
                    # df = pd.DataFrame(target_model_res[sel]).T
                    # markers, caps, bars = ax.errorbar(
                    #     df.index,
                    #     df["mean"],
                    #     label=sel,
                    #     yerr=df["std"],
                    #     fmt='-o',
                    #     capsize=3
                    # )
                    # [bar.set_alpha(0.1) for bar in bars]
                    # [cap.set_alpha(1.0) for cap in caps]
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
            save_path = self.get_path(
                        folder_name=feature_suffix,
                        inner_folders=["plots", target_suffix],
                        file_name=f"{metamodel}.png"
                    )
            if save_path.exists() and not self.rewrite:
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
