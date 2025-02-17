import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

from ms.handler.metadata_source import MetadataSource
from ms.handler.selector_handler import SelectorHandler
from ms.utils.typing import NDArrayFloatT


class TESelector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "te"

    @property
    def class_name(self) -> str:
        return "treatment_effect"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            quantile_value: float = 0.8,
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.quantile_value = quantile_value

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        inf_df = pd.DataFrame()
        for i, f_name in enumerate(features_names):
            t = x[:, i]
            covariates = np.concatenate([x[:, :i], x[:, i + 1:]], axis=1)
            model_y = RandomForestRegressor()
            model_t = RandomForestRegressor()
            dml = CausalForestDML(model_y=model_y, model_t=model_t)
            dml.fit(Y=y, T=t, X=covariates)
            te = dml.effect(X=covariates)
            inf_df[f_name] = te
        inf_df = pd.DataFrame(inf_df.mean(), index=features_names, columns=["eff_mean"])
        quantile_eff = inf_df["eff_mean"].abs().quantile(self.quantile_value)

        for i, eff in enumerate(inf_df["eff_mean"].to_numpy()):
            if abs(eff) < quantile_eff:
                inf_df.iloc[i, 0] = None

        return inf_df
