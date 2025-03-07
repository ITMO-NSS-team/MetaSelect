import random

import numpy as np

from ms.metadataset.metadata_sampler import DataSampler
from ms.pipeline.pipeline_constants import *

np.random.seed(seed)
random.seed(seed)

f_sampler = DataSampler(
        md_source=md_source,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False
    )

meta_learner = MetaLearner(
        md_source=md_source,
        opt_scoring=grid_scoring,
        model_scoring=model_scoring,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        opt_method=None,
        opt_cv=5,
        model_cv=10,
        n_trials=50,
        test_mode=False,
    )

selectors_to_use = ["base", "corr", "f_val", "mi", "xgb", "lasso", "rfe", "te", "cf"]
selectors = [all_handlers[selector][1] for selector in selectors_to_use if selector != "rfe"]
metrics_suffixes = ["perf_abs", "perf_rel", "diff"]
features_suffixes = ["power"]

if __name__ == "__main__":
    f_sampler.split_data(
        feature_suffixes=features_suffixes,
        target_suffix="perf_abs",
        splitter=k_fold_splitter,
        to_rewrite=False,
    )

    f_sampler.slice_features(
        feature_suffixes=["power"],
        to_rewrite=False,
        n_iter=1,
        slice_sizes=None, # all dataset
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
                    to_rewrite=False,
                )

    meta_learner.run_models(
        models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
        feature_suffixes=features_suffixes,
        target_suffixes=metrics_suffixes,
        selector_names=selectors_to_use,
        to_rewrite=False,
    )
