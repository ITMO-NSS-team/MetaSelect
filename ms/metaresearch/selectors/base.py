import numpy as np
import pandas as pd

from ms.handler.metadata_source import MetadataSource
from ms.handler.selector_handler import SelectorHandler
from ms.utils.typing import NDArrayFloatT


class BaseSelector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "base"

    @property
    def class_name(self) -> str:
        return "base"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        res = pd.DataFrame(index=features_names)
        res["no_selection"] = np.zeros(shape=(len(features_names),))
        return res
