import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

from ms.handler.metadata_source import MetadataSource
from ms.handler.selector_handler import SelectorHandler
from ms.utils.typing import NDArrayFloatT


class CorrelationSelector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "corr"

    @property
    def class_name(self) -> str:
        return "correlation"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            corr_type: str = "spearman",
            p_threshold: float = 0.05,
            abs_threshold: float = 0.2,
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.corr_type = corr_type
        self.p_threshold = p_threshold
        self.abs_threshold = abs_threshold

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        if self.corr_type == "pearson":
            corr_type = "pearson"
            result = pearsonr(x=x, y=y)
            stats, p_values = result.statistic[:-1, -1], result.pvalue[:-1, -1]
        else:
            corr_type = "spearman"
            result = spearmanr(a=x, b=y)
            stats, p_values = result.statistic[:-1, -1], result.pvalue[:-1, -1]

        res_df = pd.DataFrame(index=features_names)
        res_df[f"corr_{corr_type}"] = stats

        for i, p_value in enumerate(p_values):
            if p_value > self.p_threshold or abs(res_df.iloc[i, 0]) < self.abs_threshold:
                res_df.iloc[i, 0] = None

        return res_df


class Chi2Selector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "chi2"

    @property
    def class_name(self) -> str:
        return "chi2"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            p_threshold: float = 0.05,
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.p_threshold = p_threshold

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        chi2_stats, p_values = chi2(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["chi2"] = chi2_stats

        for i, p_value in enumerate(p_values):
            if p_value > self.p_threshold:
                res_df.iloc[i, 0] = None

        return res_df


class MutualInfoSelector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "mi"

    @property
    def class_name(self) -> str:
        return "mutual_info"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            quantile_value: float = 0.9,
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
        mi = mutual_info_classif(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["mi"] = mi
        quantile_mi = res_df["mi"].quantile(self.quantile_value)

        for i, mi_value in enumerate(mi):
            if mi_value < quantile_mi:
                res_df.iloc[i, 0] = None

        return res_df


class FValueSelector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "f_val"

    @property
    def class_name(self) -> str:
        return "f_value"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            p_threshold: float = 0.05,
            quantile_value: float = 0.5,
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.p_threshold = p_threshold
        self.quantile_value = quantile_value

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        f_statistic, p_values = f_classif(X=x, y=y)
        res_df = pd.DataFrame(index=features_names)
        res_df["f"] = f_statistic
        quantile_f = pd.Series(f_statistic).quantile(self.quantile_value)

        for i, p_value in enumerate(p_values):
            if p_value > self.p_threshold or abs(res_df.iloc[i, 0]) < quantile_f:
                res_df.iloc[i, 0] = None
        return res_df
