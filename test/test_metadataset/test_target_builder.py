import pandas as pd
import pytest

from ms.metadataset.metadata_source import TabzillaSource
from ms.metadataset.target_builder import TargetPerfBuilder, TargetRawBuilder
from ms.utils.calc_utils import AbsRangeStorage


def test_target_perf_builder():
    abs_ranges = AbsRangeStorage(default_init=False)
    abs_ranges.add_range(key=0, right_value=0.0, left_value=0.8)
    abs_ranges.add_range(key=1, right_value=0.8, left_value=1.0)

    target_builder = TargetPerfBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="AUC__val",
        perf_type="abs",
        abs_ranges=abs_ranges
    )
    target_builder.test_mode = True
    target_path = target_builder.handle_metrics()
    assert pd.read_csv(target_path) is not None

def test_target_raw_builder():
    target_builder = TargetRawBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="AUC__val",
    )
    target_builder.test_mode = True
    target_path = target_builder.handle_metrics()
    assert pd.read_csv(target_path) is not None

if __name__ == "__main__":
    pytest.main()