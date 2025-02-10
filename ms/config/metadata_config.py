from dataclasses import dataclass
from ms.utils.ms_utils import get_project_path, pjoin


@dataclass
class MetadataConfig:
    data_path: str = pjoin(get_project_path(), "resources")
    results_path: str = pjoin(get_project_path(), "results")
    raw_folder: str = "raw"

    formatted_folder: str = "formatted"
    filtered_folder: str = "filtered"
    preprocessed_folder: str = "preprocessed"
    target_folder: str = "target"

    model_free_folder: str = "model_free"
    correlation_folder: str = "correlation"
    info_folder: str = "info"
    chi_square_folder: str = "chi_square"

    ml_folder: str = "ml"

    features_prefix: str = "features"
    metrics_prefix: str = "metrics"
