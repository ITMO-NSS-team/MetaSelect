from collections.abc import Callable

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

from ms.handler.metadata_source import MetadataSource
from ms.handler.method_handler import SelectorHandler
from ms.utils.typing import NDArrayFloatT


class ModelFreeSelector(SelectorHandler):
    @property
    def methods(self) -> dict[str, Callable]:
        return {
            "f_value": self.__f_value_handler__,
            "mi": self.__mi_handler__,
            "chi2": self.__chi2_handler__,
            "corr": self.__correlation_handler__,
        }

    @property
    def class_name(self) -> str:
        return "model_free"

    @property
    def class_folder(self) -> str:
        return self.config.model_free_folder

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

    @staticmethod
    def __f_value_handler__(
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        f_statistic, p_values = f_classif(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["f"] = f_statistic
        # quantile_p = pd.Series(p_values).quantile(0.5)
        quantile_f = pd.Series(f_statistic).quantile(0.5)

        for i, p_value in enumerate(p_values):
            if p_value > 0.05 or abs(res_df.iloc[i, 0]) < quantile_f:
                res_df.iloc[i, 0] = None
        return res_df

    @staticmethod
    def __mi_handler__(
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        mi = mutual_info_classif(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["mi"] = mi
        quantile_mi = res_df["mi"].quantile(0.9)

        for i, mi_value in enumerate(mi):
            if mi_value < quantile_mi:
                res_df.iloc[i, 0] = None

        return res_df

    @staticmethod
    def __chi2_handler__(
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        chi2_stats, p_values = chi2(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["chi2"] = chi2_stats

        for i, p_value in enumerate(p_values):
            if p_value > 0.05:
                res_df.iloc[i, 0] = None

        return res_df

    @staticmethod
    def __correlation_handler__(
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
            method_config: dict | None = None,
    ) -> pd.DataFrame:
        if method_config is None or method_config["corr_type"] == "pearson":
            corr_type = "pearson"
            result = pearsonr(x=x, y=y)
            stats, p_values = result.statistic[:-1, -1], result.pvalue[:-1, -1]
        else:
            corr_type = "spearman"
            result = spearmanr(a=x, b=y)
            stats, p_values = result.statistic[:-1, -1], result.pvalue[:-1, -1]

        res_df = pd.DataFrame(index=features_names)
        res_df[f"corr_{corr_type}"] = stats
        # quantile_p = pd.Series(p_values).quantile(0.5)

        for i, p_value in enumerate(p_values):
            if p_value > 0.05 or abs(res_df.iloc[i, 0]) < 0.2:
                res_df.iloc[i, 0] = None

        return res_df
