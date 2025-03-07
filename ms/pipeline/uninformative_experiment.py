import random

import numpy as np

from ms.metadataset.feature_sampler import FeatureCrafter
from ms.metadataset.data_splitter import DataSampler
from ms.pipeline.pipeline_constants import *

np.random.seed(seed)
random.seed(seed)

random_crafter = FeatureCrafter(
        md_source=md_source,
        features_folder="filtered",
        metrics_folder="target",
        test_mode=False,
)

sampler = DataSampler(
        md_source=md_source,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False,
)

selectors_to_use = ["base", "corr", "f_val", "mi", "xgb", "lasso", "rfe", "te", "cf"]
selectors = [all_handlers[selector][1] for selector in selectors_to_use if selector != "rfe"]
metrics_suffixes = ["perf_abs", "perf_rel", "diff"]
features_suffixes = ["noise", "corrupted", "so"]

if __name__ == "__main__":
        random_crafter.perform(
                features_suffix=data_transform,
                random_percent=1.0, # 100% of data
                dist_name="normal",
        )
        random_crafter.perform(
                features_suffix=data_transform,
                corrupted_percent=1.0, # 100% of data
                corrupt_coeff=0.5,
        )
        random_crafter.perform(
                features_suffix=data_transform,
                second_order_percent=1.0, # 100% of data
        )
        sampler.sample_uninformative(
                feature_suffixes=features_suffixes,
                to_rewrite=True,
                n_iter=1,
                percents=[0.1, 0.3, 0.5, 0.7, 1.0] # data + % of data
        )
        sampler.split_data(
                feature_suffixes=features_suffixes,
                target_suffix="perf_abs",
                splitter=k_fold_splitter,
                to_rewrite=True,
        )

        for features_suffix in features_suffixes:
                print(features_suffix)
                for metrics_suffix in metrics_suffixes:
                        print(metrics_suffix)
                        for selector in selectors:
                                print(selector.class_name)
                                selector.perform(
                                        features_suffix=features_suffix,
                                        metrics_suffix=metrics_suffix,
                                        to_rewrite=True,
                                )

        meta_learner.run_models(
                models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
                feature_suffixes=features_suffixes,
                target_suffixes=metrics_suffixes,
                selector_names=selectors_to_use,
                to_rewrite=False,
        )
