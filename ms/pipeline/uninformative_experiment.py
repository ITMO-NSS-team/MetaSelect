from ms.metadataset.feature_engineering import FeatureCrafter
from ms.metadataset.metadata_sampler import MetadataSampler
from ms.pipeline.pipeline_constants import *

random_crafter = FeatureCrafter(
        md_source=md_source,
        features_folder="filtered",
        metrics_folder="target",
        test_mode=False,
)

sampler = MetadataSampler(
        md_source=md_source,
        splitter=train_test_slitter,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False,
)

if __name__ == "__main__":
        # random_crafter.perform(
        #         features_suffix=data_transform,
        #         random_percent=1.0,
        #         dist_name="normal",
        # )
        # random_crafter.perform(
        #         features_suffix=data_transform,
        #         corrupted_percent=1.0,
        #         corrupt_coeff=0.5,
        # )
        # random_crafter.perform(
        #         features_suffix=data_transform,
        #         second_order_percent=1.0,
        # )
        sampler.sample_data(
                feature_suffixes=["noise", "corrupted", "so"],
                target_suffix="perf_abs",
                rewrite=True,
                are_additional=True,
                percents=[0.1, 0.3, 0.5, 0.7, 1.0]
        )
