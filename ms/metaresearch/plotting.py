import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from ms.handler.metadata_handler import MetadataHandler
from ms.handler.metadata_source import MetadataSource
from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selector_data import SelectorData
from ms.utils.navigation import pjoin


class Plotter(MetadataHandler):
    @property
    def class_name(self) -> str:
        return "plotter"

    @property
    def class_folder(self) -> str:
        return self.config.plots

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

    @property
    def load_path(self) -> str:
        return self.config.results_path

    def __init__(
            self,
            md_source: MetadataSource,
            mean_cols: list[str],
            std_cols: list[str],
            names: list[str],
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
        self.names = names
        self.rewrite = rewrite

    def plot(
            self,
            models: list[MetaModel],
            target_suffixes: list[str],
            selectors: list[SelectorData],
            target_models: list[str],
    ):
        for selector in selectors:
            for target_suffix in target_suffixes:
                res_list = []
                metamodels_plot = self.get_save_path(
                    folder_name=selector.name,
                    inner_folders=[target_suffix],
                    file_name="agg.png",
                )
                for model in models:
                    load_path = self.get_save_path(
                        folder_name=selector.name,
                        inner_folders=[target_suffix],
                        file_name=self.get_file_name(
                            prefix=self.config.results_prefix,
                            suffix=model.name
                        ),
                    )
                    res = self.load(
                        folder_name="",
                        file_name="",
                        path=load_path,
                    )
                    res_list.append(res)
                    plot_path = Path(
                        pjoin(
                            Path(load_path).parent,
                            self.config.plots,
                            f"{model.name}.png"
                        )
                    )
                    if plot_path.exists() and not self.rewrite:
                        continue
                    self.plot_metamodel(
                        res=res,
                        save_path=plot_path,
                        title=f"{model.display_name} classification results",
                        target_models=target_models
                    )
                if metamodels_plot.exists() and not self.rewrite:
                    continue
                self.plot_metamodels(
                    res_list=res_list,
                    metamodels=models,
                    save_path=metamodels_plot,
                    target_models=target_models
                )


    def plot_metamodel(
            self,
            res: pd.DataFrame,
            save_path: Path,
            title: str,
            target_models: list[str]
    ) -> None:
        os.makedirs(save_path.parent, exist_ok=True)
        fig, axs = plt.subplots(2, 2)
        cells = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        for i, cell in enumerate(cells):
            cell.errorbar(
                x=target_models,
                y=res[self.mean_cols[i]],
                yerr=res[self.std_cols[i]],
                fmt='o',
                capsize=3
            )
            cell.set(ylabel=self.names[i])
        fig.tight_layout(pad=1.0)
        fig.suptitle(title)
        plt.subplots_adjust(top=0.9)

        plt.savefig(save_path)

    def plot_metamodels(
            self,
            res_list: list[pd.DataFrame],
            metamodels: list[MetaModel],
            save_path: Path,
            target_models: list[str]
    ) -> None:
        os.makedirs(save_path.parent, exist_ok=True)
        if len(target_models) > 1:
            plot_type = "line"
            size = (15, 10)
        else:
            plot_type = "bar"
            size = (9, 7)
        fig, axs = plt.subplots(2, 2, figsize=size)
        cells = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        for i, cell in enumerate(cells):
            concat_res = pd.concat(
                [res[[self.mean_cols[i]]] for res in res_list],
                axis=1
            )
            concat_res.columns = [model.name for model in metamodels]
            concat_res.index = target_models
            concat_res.plot(
                kind=plot_type,
                ax=cell,
            )
            cell.set(ylabel=self.names[i])
        fig.tight_layout(pad=1.0)

        plt.savefig(save_path)
