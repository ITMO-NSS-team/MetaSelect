from typing import Callable

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier

from ms.handler.metadata_source import MetadataSource
from ms.handler.selector_handler import SelectorHandler
from ms.utils.typing import NDArrayFloatT


class MLSelector(SelectorHandler):
    @property
    def methods(self) -> dict[str, Callable]:
        return {
            "xgb": self.__xgb_handler__,
            "lasso": self.__lasso_handler__,
            "rfe": self.__rfe__handler__,
            "causal": self.__causal_handler__,
        }

    @property
    def params(self) -> dict[str, dict]:
        return {
            "xgb": {
                "eval_metric": "merror",
                "learning_rate": 0.01,
                "max_depth": 3,
                "n_estimators": 5
            },
            "lasso": {
                "cv": 5,
                "n_alphas": 100,
            },
            "rfe": {
                "estimator": RandomForestClassifier(),
                "step": 0.9,
                "cv": 5,
            }
        }

    @property
    def class_name(self) -> str:
        return "ml"

    @property
    def class_folder(self) -> str:
        return self.config.ml_folder

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

    def __xgb_handler__(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        xgb = XGBClassifier()
        xgb.set_params(**self.params["xgb"])

        xgb.fit(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["xgb_fi"] = xgb.feature_importances_

        for i, fi in enumerate(xgb.feature_importances_):
            if fi == 0.0:
                res_df.iloc[i, 0] = None

        return res_df

    def __lasso_handler__(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        lasso = LassoCV()
        lasso.set_params(**self.params["lasso"])

        lasso.fit(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["lasso_fi"] = lasso.coef_

        for i, coef in enumerate(lasso.coef_):
            if coef == 0.0:
                res_df.iloc[i, 0] = None

        return res_df

    def __rfe__handler__(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        rfe = RFECV(**self.params["rfe"])
        rfe.fit(X=x, y=y)

        res_df = pd.DataFrame(index=features_names)
        res_df["rfe_fi"] = rfe.ranking_

        for i, rank in enumerate(rfe.ranking_):
            if rank > 1:
                res_df.iloc[i, 0] = None

        return res_df

    def __causal_handler__(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        inf_df = pd.DataFrame()
        for i, f_name in enumerate(features_names):
            T = x[:, i]
            covariates = np.concatenate([x[:, :i], x[:, i+1:]], axis=1)
            model_y = RandomForestRegressor()
            model_t = RandomForestRegressor()
            dml = CausalForestDML(model_y=model_y, model_t=model_t)
            dml.fit(Y=y, T=T, X=covariates)
            te = dml.effect(X=covariates)
            inf_df[f_name] = te
            # p = np.percentile(a=T, q=50)
        inf_df = pd.DataFrame(inf_df.mean(), index=features_names, columns=["eff_mean"])
        quantile_eff = inf_df["eff_mean"].abs().quantile(0.8)

        for i, eff in enumerate(inf_df["eff_mean"].to_numpy()):
            if abs(eff) < quantile_eff:
                inf_df.iloc[i, 0] = None

        return inf_df
