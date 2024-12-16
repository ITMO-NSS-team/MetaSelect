import pandas as pd

from sb.meta_dataset.features_filter import FeaturesFilter
from sb.meta_dataset.target_builder import TargetBuilder
from sb.utils.sb_utils import pjoin


class MetaDataset:
    def __init__(
            self,
            features_path: str,
            eval_path: str,
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            save_path: str | None = None,
    ):
        self.features_dataset = pd.read_csv(features_path, index_col=0)
        self.eval_dataset = pd.read_csv(eval_path, index_col=0)
        self.index_name = index_name
        self.alg_name = alg_name
        self.save_path = save_path

    def get_target(
            self,
            target_builder: TargetBuilder
    ) -> pd.DataFrame:
        target_data = {
            self.index_name: [],
            target_builder.name: [],
        }
        for dataset in self.eval_dataset.index.unique():
            dataset_slice = self.eval_dataset.loc[dataset, :]
            algs = dataset_slice[self.alg_name].tolist()
            if target_builder.model_1 in algs and target_builder.model_2 in algs:
                model_1_res = dataset_slice.loc[
                    dataset_slice[self.alg_name] == target_builder.model_1
                ][target_builder.metric].item()
                model_2_res = dataset_slice.loc[
                    dataset_slice[self.alg_name] == target_builder.model_2
                ][target_builder.metric].item()
                value = target_builder.get_result(
                    a=model_1_res, b=model_2_res
                )

                target_data[self.index_name].append(dataset)
                target_data[target_builder.name].append(value)

        target_dataset = pd.DataFrame(data=target_data)
        target_dataset.set_index(self.index_name, inplace=True)

        if self.save_path is not None:
            target_dataset.to_csv(
                pjoin(self.save_path, f"{target_builder.name}.csv"),
                index=False
            )

        return target_dataset

    def get_features(
            self,
            features_filter: FeaturesFilter,
    ) -> pd.DataFrame:
        filtered_features, num_features = features_filter.filter(
            features_dataset=self.features_dataset
        )
        if self.save_path is not None:
            filtered_features.to_csv(
                pjoin(self.save_path, f"features_{num_features}.csv"),
                index=False
            )
        return filtered_features

    def get_metadata(
            self,
            target_builder: TargetBuilder | None = None,
            features_filter: FeaturesFilter | None = None,
    ) -> pd.DataFrame:
        if target_builder is None:
            target_df = self.eval_dataset.copy()
        else:
            target_df = self.get_target(target_builder=target_builder)

        if features_filter is None:
            features_df = self.features_dataset.copy()
        else:
            features_df = self.get_features(features_filter=features_filter)

        target_datasets = set(target_df.index)
        features_datasets = set(features_df.index)
        intersection = list(target_datasets.intersection(features_datasets))
        return target_df.loc[intersection, :].to_frame().merge(
            features_df.loc[intersection, :],
            how='left',
            on=self.index_name
        )
