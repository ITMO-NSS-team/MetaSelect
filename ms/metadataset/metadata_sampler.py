from random import sample

import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit

from ms.handler.metadata_handler import MetadataHandler
from ms.handler.metadata_source import MetadataSource, TabzillaSource


class MetadataSampler(MetadataHandler):
    @property
    def class_name(self) -> str:
        return "features_sampler"

    @property
    def class_folder(self) -> str:
        return self.config.sampler_folder

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
        return self.config.resources

    def __init__(
            self,
            md_source: MetadataSource,
            splitter: KFold | ShuffleSplit,
            start: int = 20,
            step: int = 20,
            n_iter: int = 5,
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
        self.splitter = splitter
        self.start = start
        self.step = step
        self.n_iter = n_iter

    def sample_data(
            self,
            feature_suffixes: list[str],
            target_suffix: str,
            rewrite: bool = False,
            are_additional: bool = False,
            percents: list[float] | None = None,
    ) -> None:
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
            y_df = self.load_metrics(suffix=target_suffix)

            data_split = self.splitter.split(x_df, y_df)

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

            if are_additional:
                self.sample_uninformative(
                    add_suffix=feature_suffix,
                    percents=percents,
                    rewrite=rewrite,
                    x_df=x_df
                )
            else:
                self.make_samples(
                    processed_suffix=feature_suffix,
                    feature_dataset=x_df,
                    rewrite=rewrite,
                )


    def make_samples(
            self,
            processed_suffix: str,
            feature_dataset: pd.DataFrame,
            rewrite: bool = False,
    ) -> None:
        save_path = self.get_path(
            folder_name=self.config.sampler_folder,
            file_name=f"{self.config.slices_prefix}.json",
            inner_folders=[processed_suffix],
        )
        if not rewrite and save_path.exists():
            return
        samples_dict = {}

        f_num = feature_dataset.shape[1]
        samples_range = list(range(self.start, f_num, self.step))

        f_cols = [i for i in range(f_num)]
        if samples_range[-1] != f_num:
            samples_range.append(f_num)

        for n_samples in samples_range:
            samples_dict[n_samples] = {}
            for i in range(self.n_iter):
                samples = sample(f_cols, n_samples)
                samples_dict[n_samples][i] = samples

        self.save_samples(
            data=samples_dict,
            file_name=f"{self.config.slices_prefix}",
            inner_folders=[processed_suffix]
        )

    def sample_uninformative(
            self,
            x_df: pd.DataFrame,
            add_suffix: str,
            percents: list[float],
            rewrite: bool = False,
    ) -> None:
        save_path = self.get_path(
            folder_name=self.config.sampler_folder,
            file_name=f"{self.config.addition_prefix}.json",
            inner_folders=[add_suffix],
        )
        if not rewrite and save_path.exists():
            return
        samples_dict = {}
        additional_indexes = []
        for i, f in enumerate(list(x_df.columns)):
            if f.split("___")[0] == add_suffix:
                additional_indexes.append(i)

        for i, percent in enumerate(percents):
            sample_size = int(len(additional_indexes) * percent)
            samples_dict[i] = {}
            for j in range (self.n_iter):
                samples_dict[i][j] = sample(additional_indexes, sample_size)

        self.save_samples(
            data=samples_dict,
            file_name=f"{self.config.addition_prefix}",
            inner_folders=[add_suffix]
        )


if __name__ == "__main__":
    k_fold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    train_test_slitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    f_sampler = MetadataSampler(
        md_source=TabzillaSource(),
        splitter=k_fold_splitter,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False
    )
    f_sampler.sample_data(
        feature_suffixes=["power"],
        target_suffix="perf_abs",
        rewrite=True
    )
