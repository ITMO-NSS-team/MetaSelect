from abc import abstractmethod, ABC

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier

from ms.handler.metadata_source import MetadataSource
from ms.handler.selector_handler import SelectorHandler
from ms.utils.typing import NDArrayFloatT


class ModelBased(ABC):
    @property
    @abstractmethod
    def params(self) -> dict:
        ...

class XGBSelector(SelectorHandler, ModelBased):
    @property
    def class_folder(self) -> str:
        return "xgb"

    @property
    def class_name(self) -> str:
        return "xgb"

    @property
    def params(self) -> dict:
        return {
                "eval_metric": "merror",
                "learning_rate": 0.01,
                "max_depth": 3,
                "n_estimators": 5
            }

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            importance_threshold: float = 0.0,
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.importance_threshold = importance_threshold

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        xgb = XGBClassifier()
        xgb.set_params(**self.params)

        xgb.fit(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["xgb_fi"] = xgb.feature_importances_

        for i, fi in enumerate(xgb.feature_importances_):
            if abs(fi) <= self.importance_threshold:
                res_df.iloc[i, 0] = None

        return res_df


class LassoSelector(SelectorHandler, ModelBased):
    @property
    def class_folder(self) -> str:
        return "lasso"

    @property
    def class_name(self) -> str:
        return "lasso"

    @property
    def params(self) -> dict:
        return {
                "cv": 5,
                "n_alphas": 100,
            }

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            coef_threshold: float = 0.0,
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.coef_threshold = coef_threshold

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        lasso = LassoCV()
        lasso.set_params(**self.params)

        lasso.fit(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["lasso_fi"] = lasso.coef_

        for i, coef in enumerate(lasso.coef_):
            if abs(coef) <= self.coef_threshold:
                res_df.iloc[i, 0] = None

        return res_df


class RFESelector(SelectorHandler, ModelBased):
    @property
    def class_folder(self) -> str:
        return "rfe"

    @property
    def class_name(self) -> str:
        return "rfe"

    @property
    def params(self) -> dict:
        return {
                "estimator": RandomForestClassifier(),
                "step": 0.9,
                "cv": 5,
            }

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            rank_threshold: float = 1.0,
            test_mode: bool = False,
    ) -> None:
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
        rfe = RFECV(**self.params)
        rfe.fit(X=x, y=y)

        res_df = pd.DataFrame(index=features_names)
        res_df["rfe_fi"] = rfe.ranking_

        for i, rank in enumerate(rfe.ranking_):
            if rank > self.rank_threshold:
                res_df.iloc[i, 0] = None

        return res_df
