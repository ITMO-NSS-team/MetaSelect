from ms.metaresearch.plotting import Plotter
from ms.pipeline.init_features_experiment import metrics_suffixes, selectors_to_use
from ms.pipeline.pipeline_constants import *

feature_suffixes = ["power"]

if __name__ == "__main__":
    plotter = Plotter(
        md_source=md_source,
        mean_cols=mean_cols,
        std_cols=std_cols,
    )
    selectors_loaded = meta_learner.load_selectors(
        features_suffixes=feature_suffixes,
        metrics_suffixes=metrics_suffixes,
        selector_names=selectors_to_use,
        all_data=True
    )
    plotter.plot(
        models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
        feature_suffixes=feature_suffixes,
        target_suffixes=["perf_abs", "perf_rel"],
        selectors=selectors_loaded,
        target_models=target_models,
    )
    plotter.plot(
        models=[knn_mm, lr_mm, xgb_mm, mlp_mm],
        feature_suffixes=feature_suffixes,
        target_suffixes=["diff"],
        selectors=selectors_loaded,
        target_models=["RN_XGB"],
    )
