from random import sample

import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit

from ms.handler.data_handler import DataHandler
from ms.handler.data_source import DataSource


class DataSampler(DataHandler):
    @property
    def class_name(self) -> str:
        return "features_sampler"

    @property
    def class_folder(self) -> str:
        return self.config.sampler_folder

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
        return self.config.resources

    def __init__(
            self,
            md_source: DataSource,
            # splitter: KFold | ShuffleSplit,
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
        # self.splitter = splitter

    def split_data(
            self,
            feature_suffixes: list[str],
            target_suffix: str,
            splitter: KFold | ShuffleSplit,
            rewrite: bool = False,
    ) -> None:
        y_df = self.load_metrics(suffix=target_suffix)
        for feature_suffix in feature_suffixes:
            splits_dict = {}
            splits_path = self.get_path(
                folder_name=self.config.sampler_folder,
                file_name=f"{self.config.splits_prefix}.json",
                inner_folders=[feature_suffix],
            )
            if not rewrite and splits_path.exists():
                continue

            x_df = self.load_features(suffix=feature_suffix)

            data_split = splitter.split(x_df, y_df)

            for i, (train, test) in enumerate(data_split):
                splits_dict[i] = {
                    "train": list(map(int, train)),
                    "test": list(map(int, test)),
                }

            self.save_samples(
                data=splits_dict,
                file_name=f"{self.config.splits_prefix}",
                inner_folders=[feature_suffix],
            )

    def slice_features(
            self,
            feature_suffixes: list[str],
            rewrite: bool = False,
            n_iter: int = 1,
            slice_sizes: list[int] | None = None,
    ) -> None:
        for feature_suffix in feature_suffixes:
            x_df = self.load_features(suffix=feature_suffix)
            self.make_slices(
                suffix=feature_suffix,
                x_df=x_df,
                rewrite=rewrite,
                slice_sizes=slice_sizes,
                n_iter=n_iter,
            )

    def slice_additional_features(
            self,
            feature_suffixes: list[str],
            rewrite: bool = False,
            n_iter: int = 1,
            percents: list[float] | None = None,
    ) -> None:
        for feature_suffix in feature_suffixes:
            x_df = self.load_features(suffix=feature_suffix)
            self.sample_uninformative(
                suffix=feature_suffix,
                percents=percents,
                rewrite=rewrite,
                x_df=x_df,
                n_iter=n_iter,
            )


    def make_slices(
            self,
            suffix: str,
            x_df: pd.DataFrame,
            rewrite: bool = False,
            slice_sizes: list[int] | None = None,
            n_iter: int = 5,
    ) -> None:
        if slice_sizes is None:
            slice_sizes = [x_df.shape[1]]
        save_path = self.get_path(
            folder_name=self.config.sampler_folder,
            file_name=f"{self.config.slices_prefix}.json",
            inner_folders=[suffix],
        )
        if not rewrite and save_path.exists():
            return
        samples_dict = {}

        f_num = x_df.shape[1]
        f_cols = [i for i in range(f_num)]

        for size in slice_sizes:
            samples_dict[size] = {}
            for i in range(n_iter):
                slice_sizes = sample(f_cols, size)
                samples_dict[size][i] = slice_sizes

        self.save_samples(
            data=samples_dict,
            file_name=f"{self.config.slices_prefix}",
            inner_folders=[suffix]
        )

    def sample_uninformative(
            self,
            suffix: str,
            x_df: pd.DataFrame,
            rewrite: bool = False,
            percents: list[float] | None = None,
            n_iter: int = 1,
    ) -> None:
        if percents is None:
            percents = [0.1, 0.5, 1.0]
        save_path = self.get_path(
            folder_name=self.config.sampler_folder,
            file_name=f"{self.config.slices_prefix}.json",
            inner_folders=[suffix],
        )
        if not rewrite and save_path.exists():
            return
        samples_dict = {}
        additional_indices = []
        original_indices = []
        for i, f in enumerate(list(x_df.columns)):
            if f.split("___")[0] == suffix:
                additional_indices.append(i)
            else:
                original_indices.append(i)

        for i, percent in enumerate(percents):
            sample_size = int(len(additional_indices) * percent)
            samples_dict[i] = {}
            for j in range (n_iter):
                samples_dict[i][j] = sample(additional_indices, sample_size) + original_indices

        self.save_samples(
            data=samples_dict,
            file_name=f"{self.config.slices_prefix}",
            inner_folders=[suffix]
        )
