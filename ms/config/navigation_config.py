from dataclasses import dataclass
from ms.utils.navigation import get_project_path, pjoin


@dataclass
class NavigationConfig:
    project_path: str = pjoin(get_project_path())
    resources: str = pjoin(project_path, "resources")
    results_path: str = pjoin(project_path, "results")
    plots_path: str = pjoin(project_path, "plots")

    raw_folder: str = "raw"
    formatted_folder: str = "formatted"
    filtered_folder: str = "filtered"
    preprocessed_folder: str = "preprocessed"
    target_folder: str = "target"
    sampler_folder: str = "sampler"
    handler_info_suffix: str = "info"

    range_name: str = "range"
    dataset_name: str = "dataset_name"
    alg_name: str = "alg_name"

    meta_learning: str = "meta_learning"
    plots: str = "plots"

    model_free_folder: str = "model_free"
    ml_folder: str = "ml"

    features_prefix: str = "features"
    metrics_prefix: str = "metrics"
    results_prefix: str = "results"
    splits_prefix: str = "splits"
    slices_prefix: str = "slices"
