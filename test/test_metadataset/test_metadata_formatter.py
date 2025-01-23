import pandas as pd
import pytest

from ms.metadataset.metadata_formatter import TabzillaFormatter


def test_tabzilla_formatter():
    formatter = TabzillaFormatter(
        features_folder="raw",
        metrics_folder="raw",
    )
    formatter.test_mode = True
    features_path = formatter.handle_features()
    metrics_path = formatter.handle_metrics()

    assert pd.read_csv(features_path) is not None
    assert pd.read_csv(metrics_path) is not None


if __name__ == "__main__":
    pytest.main()
