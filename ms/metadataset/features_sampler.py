from random import sample

import pandas as pd
from sklearn.model_selection import train_test_split

from ms.handler.metadata_handler import MetadataHandler
from ms.handler.metadata_source import MetadataSource, TabzillaSource


class FeaturesSampler(MetadataHandler):
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
            samples_range: list[int] | None = None,
            n_iter: int = 5,
            test_size: float = 0.3,
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
        self.samples_range = samples_range
        self.n_iter = n_iter
        self.test_size = test_size

    def sample_data(
            self,
            feature_suffixes: list[str],
            target_suffix: str,
    ) -> None:
        for feature_suffix in feature_suffixes:
            samples_dict = {}
            x_df = self.load_features(suffix=feature_suffix)
            y_df = self.load_metrics(suffix=target_suffix)

            x_train, x_test, y_train, y_test = train_test_split(
                    x_df, y_df, test_size=self.test_size)
            split_dict = {
                    "x_train": list(x_train.index),
                    "x_test": list(x_test.index),
                    "y_train": list(y_train.index),
                    "y_test": list(y_test.index),
            }
            self.save_samples(
                data=split_dict,
                file_name="split",
                inner_folders=[feature_suffix]
            )

            samples_range = self.get_samples_range(features_dataset=x_train)
            features_list = list(x_train.columns)
            for i in samples_range:
                samples_dict[i] = {}
                for j in range(self.n_iter):
                    samples_dict[i][j] = sample(features_list, i)
            self.save_samples(
                data=samples_dict,
                file_name="samples",
                inner_folders=[feature_suffix]
            )


    def get_samples_range(self, features_dataset: pd.DataFrame) -> list[int]:
        if self.samples_range is not None:
            return self.samples_range
        features = features_dataset.shape[1]
        samples_range = list(range(20, features, 20))
        if samples_range[-1] != features:
            samples_range.append(features)

        return samples_range

if __name__ == "__main__":
    f_sampler = FeaturesSampler(
        md_source=TabzillaSource(),
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False
    )
    f_sampler.sample_data(
        feature_suffixes=["power"],
        target_suffix="perf_abs"
    )
