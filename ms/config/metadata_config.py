from dataclasses import dataclass
from ms.utils.ms_utils import get_project_path, pjoin


@dataclass
class MetadataConfig:
    data_path: str = pjoin(get_project_path(), "resources")
    raw_folder: str = "raw"
    formatted_folder: str = "formatted"
    filtered_folder: str = "filtered"
    preprocessed_folder: str = "preprocessed"
    target_raw_folder: str = "target_raw"
    target_perf_folder: str = "target_perf"
    target_diff_folder: str = "target_diff"
    features_name: str = "metafeatures.csv"
    metrics_name: str = "metrics.csv"
