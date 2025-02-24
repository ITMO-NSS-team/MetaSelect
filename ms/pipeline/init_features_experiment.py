import random

import numpy as np

from ms.metadataset.metadata_sampler import MetadataSampler
from ms.pipeline.pipeline_constants import *

np.random.seed(seed)
random.seed(seed)

f_sampler = MetadataSampler(
        md_source=md_source,
        splitter=train_test_slitter,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False
    )

selectors_to_use = ["base", "corr", "f_val", "mi", "xgb", "lasso", "rfe", "te", "cf"]
selectors = [all_handlers[selector][1] for selector in selectors_to_use if selector != "rfe"]
metrics_suffixes = ["perf_abs", "perf_rel", "diff"]
features_suffixes = ["power"]

if __name__ == "__main__":
    f_sampler.sample_data(
        feature_suffixes=["power"],
        target_suffix="perf_abs",
        rewrite=True
    )

    for features_suffix in features_suffixes:
        print(features_suffix)
        for metrics_suffix in metrics_suffixes:
            print(metrics_suffix)
            for selector in selectors:
                print(selector.name)
                selector.perform(
                    features_suffix=features_suffix,
                    metrics_suffix=metrics_suffix,
                    rewrite=False,
                )

    meta_learner.run_models(
        models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
        feature_suffixes=features_suffixes,
        target_suffixes=metrics_suffixes,
        selector_names=selectors_to_use,
        rewrite=False,
        to_save=True,
    )
