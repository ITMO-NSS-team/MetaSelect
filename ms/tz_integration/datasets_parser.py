import json
import os
from os.path import join as pjoin

from ms.utils.sb_utils import parse_sb_config, make_path


def parse_datasets() -> dict:
    sb_config = parse_sb_config()
    datasets_path = make_path(sb_config["datasets_path"])
    datasets = os.listdir(datasets_path)
    result_datasets = {}
    for dataset in datasets:
        dataset_path = pjoin(datasets_path, dataset)
        metadata_path = pjoin(dataset_path, "metadata.json")
        with open(metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
        num_rows = metadata['num_instances']
        num_cols = metadata['num_features']
        target_type = metadata['target_type']
        if num_rows * num_cols < 500000 and target_type != "regression":
            result_datasets[dataset] = num_rows * num_cols
    return result_datasets


if __name__ == "__main__":
    print(len(parse_datasets()))