from pathlib import Path
from random import sample

import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit

from ms.handler.data_handler import DataHandler
from ms.handler.data_source import DataSource
from ms.utils.navigation import rewrite_decorator


class DataSampler(DataHandler):
    @property
    def class_suffix(self) -> str | None:
        return None

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
    def load_root(self) -> str:
        return self.config.resources

    @property
    def save_root(self) -> str:
        return self.config.resources

    def __init__(
            self,
            md_source: DataSource,
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

    def split_data(
            self,
            feature_suffixes: list[str],
            target_suffix: str,
            splitter: KFold | ShuffleSplit,
            to_rewrite: bool = False,
    ) -> None:
        y_df = self.load_metrics(suffix=target_suffix)
        for feature_suffix in feature_suffixes:
            save_path = self.get_samples_path(
                root_type="save",
                folders=[feature_suffix],
                sample_type=self.config.splits_prefix,
            )
            x_df = self.load_features(suffix=feature_suffix)
            self._split_data(
                save_path=save_path,
                x_df=x_df,
                y_df=y_df,
                splitter=splitter,
                to_rewrite=to_rewrite,
            )


    def slice_features(
            self,
            feature_suffixes: list[str],
            to_rewrite: bool = False,
            n_iter: int = 1,
            slice_sizes: list[int] | None = None,
    ) -> None:
        for feature_suffix in feature_suffixes:
            save_path = self.get_samples_path(
                root_type="save",
                folders=[feature_suffix],
                sample_type=self.config.slices_prefix,
            )
            x_df = self.load_features(suffix=feature_suffix)
            self._slice_features(
                save_path=save_path,
                x_df=x_df,
                slice_sizes=slice_sizes,
                n_iter=n_iter,
                to_rewrite=to_rewrite,
            )

    def sample_uninformative(
            self,
            feature_suffixes: list[str],
            to_rewrite: bool = False,
            n_iter: int = 1,
            percents: list[float] | None = None,
    ) -> None:
        for feature_suffix in feature_suffixes:
            save_path = self.get_samples_path(
                root_type="save",
                folders=[feature_suffix],
                sample_type=self.config.slices_prefix,
            )
            x_df = self.load_features(suffix=feature_suffix)
            self._sample_uninformative(
                save_path=save_path,
                suffix=feature_suffix,
                percents=percents,
                to_rewrite=to_rewrite,
                x_df=x_df,
                n_iter=n_iter,
            )

    @rewrite_decorator
    def _split_data(
            self,
            save_path: str,
            x_df: pd.DataFrame,
            y_df: pd.DataFrame,
            splitter: KFold | ShuffleSplit,
    ) -> None:
        splits_dict = {}

        data_split = splitter.split(x_df, y_df)

        for i, (train, test) in enumerate(data_split):
            splits_dict[i] = {
                "train": list(map(int, train)),
                "test": list(map(int, test)),
            }

        self.save_samples(
            samples=splits_dict,
            sample_type=self.config.splits_prefix,
            path=save_path,
        )

    @rewrite_decorator
    def _slice_features(
            self,
            save_path: Path,
            x_df: pd.DataFrame,
            slice_sizes: list[int] | None = None,
            n_iter: int = 5,
    ) -> None:
        if slice_sizes is None:
            slice_sizes = [x_df.shape[1]]

        samples_dict = {}

        f_num = x_df.shape[1]
        f_cols = [i for i in range(f_num)]

        for size in slice_sizes:
            samples_dict[size] = {}
            for i in range(n_iter):
                slice_sizes = sample(f_cols, size)
                samples_dict[size][i] = slice_sizes

        self.save_samples(
            samples=samples_dict,
            sample_type=self.config.slices_prefix,
            path=save_path,
        )

    @rewrite_decorator
    def _sample_uninformative(
            self,
            save_path: Path,
            suffix: str,
            x_df: pd.DataFrame,
            percents: list[float] | None = None,
            n_iter: int = 1,
    ) -> None:
        if percents is None:
            percents = [0.1, 0.5, 1.0]

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
            samples=samples_dict,
            sample_type=self.config.slices_prefix,
            path=save_path,
        )
