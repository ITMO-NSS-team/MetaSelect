import random

from ms.metaresearch.plotting import Plotter
from ms.pipeline import meta_learner, metrics_suffixes, selectors_to_use
from ms.pipeline_constants import *

np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
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
