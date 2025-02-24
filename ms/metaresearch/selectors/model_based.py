from abc import abstractmethod, ABC

import pandas as pd
from sklearn.linear_model import LogisticRegression
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
        lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',  # 'liblinear' or 'saga' for L1
            C=0.1,
            random_state=42
        )

        lasso.fit(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        lasso_coef = lasso.coef_.flatten()
        res_df["lasso_fi"] = lasso_coef

        for i, coef in enumerate(lasso_coef):
            if abs(coef) <= self.coef_threshold:
                res_df.iloc[i, 0] = None

        return res_df
