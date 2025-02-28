from abc import ABC

import pandas as pd
from sklearn.feature_selection import RFE

from ms.handler.data_source import DataSource
from ms.handler.selector_handler import SelectorHandler
from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selectors.model_based import ModelBased
from ms.utils.typing import NDArrayFloatT


class ModelWrapper(ABC):
    def __init__(
            self,
            model: MetaModel,
    ):
        self.model = model

class RFESelector(SelectorHandler, ModelBased, ModelWrapper):
    @property
    def class_folder(self) -> str:
        return "rfe"

    @property
    def class_name(self) -> str:
        return "rfe"

    @property
    def params(self) -> dict:
        return {
            "estimator": self.model.model,
        }

    def __init__(
            self,
            md_source: DataSource,
            model: MetaModel,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            rank_threshold: float = 1.0,
            test_mode: bool = False,
    ) -> None:
        self.model = model
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )

        self.rank_threshold = rank_threshold

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        rfe = RFE(**self.params)
        rfe.fit(X=x, y=y)

        res_df = pd.DataFrame(index=features_names)
        res_df["rfe_fi"] = rfe.ranking_

        for i, rank in enumerate(rfe.ranking_):
            if rank > self.rank_threshold:
                res_df.iloc[i, 0] = None

        return res_df
