import pandas as pd
import pytest

from ms.metadataset.metadata_filter import TabzillaFilter


def test_tabzilla_filter():
    md_filter = TabzillaFilter(
        features_folder="formatted",
        metrics_folder="formatted",
        keys_to_exclude=["histogram", "count"],
        models_list=["CatBoost", "XGBoost", "MLP"],
    )
    md_filter.test_mode = True
    features_path = md_filter.handle_features()
    metrics_path = md_filter.handle_metrics()

    assert pd.read_csv(features_path) is not None
    assert pd.read_csv(metrics_path) is not None


if __name__ == "__main__":
    pytest.main()