import pandas as pd
import pytest

from ms.metadataset.metadata_preprocessor import PrelimPreprocessor
from ms.metadataset.metadata_source import TabzillaSource

def test_prelim_preprocessor():


    preprocessor = PrelimPreprocessor(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="target_raw",
        perf_type="abs",
    )
    preprocessor.test_mode = True
    features_path = preprocessor.handle_features()
    metrics_path = preprocessor.handle_metrics()

    assert pd.read_csv(features_path) is not None
    assert pd.read_csv(metrics_path) is not None


if __name__ == "__main__":
    pytest.main()
