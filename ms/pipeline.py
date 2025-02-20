import random

from ms.metadataset.metadata_sampler import MetadataSampler
from ms.metaresearch.meta_learning import MetaLearner
from ms.metaresearch.plotting import Plotter
from ms.pipeline_constants import *

np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    selectors_to_use = ["base", "corr", "f_val", "mi", "xgb", "lasso", "rfe", "te"]
    selectors = [all_handlers[selector][1] for selector in selectors_to_use if selector != "rfe"]
    metrics_suffixes = ["perf_abs", "perf_rel", "diff"]

    f_sampler = MetadataSampler(
        md_source=md_source,
        splitter=train_test_slitter,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False
    )
    f_sampler.sample_data(
        feature_suffixes=["power"],
        target_suffix="perf_abs",
        rewrite=False
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
    meta_learner.run_models(
        models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
        feature_suffixes=features_suffixes,
        target_suffixes=metrics_suffixes,
        selector_names=selectors_to_use,
        rewrite=False,
        to_save=True,
    )

    plotter = Plotter(
        md_source=md_source,
        mean_cols=mean_cols,
        std_cols=std_cols,
        rewrite=False
    )
    selectors_loaded = meta_learner.load_selectors(
            features_suffixes=features_suffixes,
            metrics_suffixes=metrics_suffixes,
            selector_names=selectors_to_use,
            all_data=True
        )
    plotter.plot(
        models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
        feature_suffixes=features_suffixes,
        target_suffixes=["perf_abs", "perf_rel"],
        selectors=selectors_loaded,
        target_models=target_models,
    )
    plotter.plot(
        models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
        feature_suffixes=features_suffixes,
        target_suffixes=["diff"],
        selectors=selectors_loaded,
        target_models=["RN_XGB"],
    )
