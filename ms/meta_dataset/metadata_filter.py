from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MetadataFilter(ABC):
    name: str = "base_filter"
    def __init__(self):
        pass

    @abstractmethod
    def filter_features(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def filter_target(self, target_dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class TabzillaFilter(MetadataFilter):
    name = "tabzilla_filter"

    def __init__(
            self,
            dataset_name: str = "dataset_name",
            round_attrs: list[str] | None = None,
            filter_families: list[str] | None = None,
            nan_threshold: float = 0.5,
            features_to_exclude: list[str] | None = None,
            keys_to_exclude: list[str] | None = None,
            datasets_to_exclude: list[str] | None = None,
            models_to_exclude: list[str] | None = None,
            models_priority_list: list[str] | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.round_attrs = round_attrs if round_attrs is not None else \
        [
            "f__pymfe.general.nr_inst",
            "f__pymfe.general.nr_attr",
            "f__pymfe.general.nr_bin",
            "f__pymfe.general.nr_cat",
            "f__pymfe.general.nr_num",
            "f__pymfe.general.nr_class",
        ]
        self.filter_families = filter_families
        self.nan_threshold = nan_threshold
        self.features_to_exclude = features_to_exclude
        self.keys_to_exclude = keys_to_exclude
        self.datasets_to_exclude = datasets_to_exclude
        self.models_to_exclude = models_to_exclude
        self.models_priority_list = models_priority_list

    def filter_features(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        agg_features = features_dataset.groupby(self.dataset_name).median(numeric_only=True)
        for attr in self.round_attrs:
            agg_features.loc[:, attr] = agg_features[attr].round(0)

        num_datasets = len(agg_features.index)
        for col in agg_features:
            x = agg_features[col].to_numpy()
            if (agg_features[col].isna().sum() > num_datasets * self.nan_threshold
                    or np.all(x == x[0])):
                agg_features.drop(col, axis="columns", inplace=True)

        agg_features = agg_features.fillna(
            agg_features.median(numeric_only=True)
        )
        if self.filter_families is not None:
            prefixes = [f"f__pymfe.{family}" for family in self.filter_families]
            filter_cols = [
                col
                for col in agg_features.columns
                if not col.startswith("f__")
                   or any(col.startswith(prefix) for prefix in prefixes)
            ]
            agg_features = agg_features[filter_cols]

        agg_features.drop(self.features_to_exclude, axis="columns", inplace=True)
        agg_features.drop(self.datasets_to_exclude, axis="index", inplace=True)
        if self.keys_to_exclude is not None:
            new_features = []
            for f in agg_features.col:
                is_included = True
                for key in self.keys_to_exclude:
                    if key in f:
                        is_included = False
                        break
                if is_included:
                    new_features.append(f)
            agg_features = agg_features.loc[:, new_features]

        return agg_features

    def filter_target(self, target_dataset: pd.DataFrame) -> pd.DataFrame:
        res_target = target_dataset.copy()
        res_target.drop(self.models_to_exclude, axis="columns", inplace=True)
        priority_list = self.models_priority_list if self.models_priority_list is not None \
            else list(res_target.columns)
        for model in priority_list:
            for row in res_target.index:
                if np.isnan(res_target.loc[row, model]):
                    res_target.drop(row, axis="rows", inplace=True)

        return res_target