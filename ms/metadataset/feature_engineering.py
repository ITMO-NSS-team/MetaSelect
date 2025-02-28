from random import sample

import numpy as np

from ms.handler.data_handler import DataHandler
from ms.handler.data_source import DataSource
from ms.utils.typing import NDArrayFloatT


class FeatureCrafter(DataHandler):
    distribution = {
        "normal": np.random.normal,
        "uniform": np.random.uniform,
        "poisson": np.random.poisson,
        "gamma": np.random.gamma,
    }

    @property
    def source(self) -> DataSource:
        return self._md_source

    @property
    def class_name(self) -> str:
        return "crafter"

    @property
    def has_index(self) -> dict:
        return {
            "features": True,
            "metrics": True,
        }

    @property
    def save_path(self) -> str:
        return self.config.resources

    @property
    def load_path(self) -> str:
        return self.config.resources

    @property
    def class_folder(self) -> str:
        return self.config.preprocessed_folder

    def __init__(
            self,
            md_source: DataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "target",
            test_mode: bool = False,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source

    def perform(
            self,
            features_suffix: str,
            random_percent: float | None = None,
            corrupted_percent: float | None = None,
            second_order_percent: float | None = None,
            dist_name: str = "normal",
            corrupt_coeff: float = 0.5,
    ) -> None:
        processed_dataset = self.load(
            folder_name="preprocessed",
            file_name=self.get_file_name(self.config.features_prefix, features_suffix),
        )
        cols = list(processed_dataset.columns)
        rows = list(processed_dataset.index)
        features = len(cols)
        datasets = len(rows)
        features_dataset = self.load_features().loc[rows, cols]
        changed_dataset = processed_dataset.copy()
        if random_percent is not None:
            r_num = int(features * random_percent)
            for i in range(r_num):
                changed_dataset[f"noise___{dist_name}_{i}"] = self.add_random_feature(
                    size=datasets,
                    dist_name=dist_name
                )

        if corrupted_percent is not None:
            c_num = int(features * corrupted_percent)
            sampled = sample(cols, c_num)
            for f_name in sampled:
                feature = features_dataset.loc[:, f_name].to_numpy(copy=True)
                changed_dataset[f"corrupted___{f_name}"] \
                    = self.add_corrupted_feature(
                    feature=feature,
                    corrupt_coeff=corrupt_coeff,
                    dist_name=dist_name
                )

        if second_order_percent is not None:
            so_num = int(features * second_order_percent)
            for i in range(so_num):
                f_name1, f_name2 = sample(cols, 2)
                feature1 = features_dataset.loc[:, f_name1].to_numpy(copy=True)
                feature2 = features_dataset.loc[:, f_name2].to_numpy(copy=True)
                changed_dataset[f"so___{f_name1}_{f_name2}"] \
                    = self.add_second_order_feature(feature_first=feature1, feature_second=feature2)

        percents = [random_percent, corrupted_percent, second_order_percent]
        names = ["noise", "corrupted", "so"]
        save_suffix = ""
        for i, percent in enumerate(percents):
            if percent is not None:
                save_suffix += f"{names[i]}"

        self.save_features(changed_dataset, save_suffix)

    def add_random_feature(
            self,
            size: int,
            dist_name: str
    ) -> NDArrayFloatT:
        return self.distribution[dist_name](size=size)

    def add_corrupted_feature(
            self,
            feature: NDArrayFloatT,
            corrupt_coeff: float,
            dist_name: str,
    ) -> NDArrayFloatT:
        return (feature * corrupt_coeff
                + self.distribution[dist_name](size=feature.shape) * (1 - corrupt_coeff))

    @staticmethod
    def add_second_order_feature(
            feature_first: NDArrayFloatT,
            feature_second: NDArrayFloatT
    ) -> NDArrayFloatT:
        return feature_first * feature_second
