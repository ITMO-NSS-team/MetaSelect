from random import sample

import pandas as pd

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

    def sample_features(self, feature_suffixes: list[str]) -> None:
        for feature_suffix in feature_suffixes:
            samples_dict = {}
            x_df = self.load_features(suffix=feature_suffix)
            samples_range = self.get_samples_range(features_dataset=x_df)
            features_list = list(x_df.columns)
            for i in range(self.n_iter):
                samples_dict[i] = {}
                for num in samples_range:
                    samples_dict[i][num] = sample(features_list, num)
            self.save_samples(
                data=samples_dict,
                suffix=feature_suffix
            )


    def get_samples_range(self, features_dataset: pd.DataFrame) -> list[int]:
        if self.samples_range is not None:
            return self.samples_range
        features = features_dataset.shape[1]
        samples_range = list(range(20, features, 10))
        samples_range.append(features)

        return samples_range

if __name__ == "__main__":
    f_sampler = FeaturesSampler(
        md_source=TabzillaSource(),
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False
    )
    f_sampler.sample_features(feature_suffixes=["power"])
