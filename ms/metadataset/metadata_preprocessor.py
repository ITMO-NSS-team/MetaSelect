from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_handler import FeaturesHandler, MetricsHandler
from ms.handler.metadata_source import MetadataSource, TabzillaSource
from ms.utils.metadata_utils import remove_constant_features


class MetadataPreprocessor(FeaturesHandler, MetricsHandler, ABC):
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "power": PowerTransformer,
        "quantile": QuantileTransformer
    }

    @property
    @abstractmethod
    def has_suffix(self) -> bool:
        pass

    @property
    def class_name(self) -> str:
        return "preprocessor"

    @property
    def class_folder(self) -> str:
        return self.config.preprocessed_folder

    @property
    def source(self) -> MetadataSource:
        return self._md_source

    @property
    def has_index(self) -> dict:
        return {
            "features": True,
            "metrics": True,
        }

    @property
    def handler_path(self) -> str:
        return self.config.data_path

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "target",
            to_scale: list[str] | None = None,
            test_mode: bool = False,
    ):
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.to_scale = to_scale if to_scale is not None else []
        self.common_datasets = list[str] | None

    def get_common_datasets(
            self,
            feature_suffix: str = None,
            metrics_suffix: str = None
    ) -> list[str]:
        features_datasets = self.load_features(suffix=feature_suffix).index
        metrics_datasets = self.load_metrics(suffix=metrics_suffix).index
        return list(set(features_datasets) & set(metrics_datasets))

    def preprocess(self, feature_suffix: str = None, metrics_suffix: str = None) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        if (self.data_folder["features"] != self.config.preprocessed_folder
                or self.data_folder["metrics"] != self.config.preprocessed_folder):
            self.common_datasets = self.get_common_datasets(
                feature_suffix=feature_suffix,
                metrics_suffix=metrics_suffix
            )
        else:
            self.common_datasets = None

        processed_features = self.handle_features(
            load_suffix=feature_suffix,
            save_suffix=None if self.has_suffix else feature_suffix,
            to_save=True
        )
        processed_metrics = self.handle_metrics(
            load_suffix=metrics_suffix,
            save_suffix=metrics_suffix,
            to_save=True
        )

        return processed_features, processed_metrics

    def __handle_features__(self, features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        if self.common_datasets is not None:
            processed_dataset = features_dataset.copy().loc[self.common_datasets].sort_index()
        else:
            processed_dataset = features_dataset.copy()
        return self.__process_features__(processed_dataset=processed_dataset)

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        if self.common_datasets is not None:
            processed_dataset = metrics_dataset.copy().loc[self.common_datasets].sort_index()
        else:
            processed_dataset = metrics_dataset.copy()
        return processed_dataset, HandlerInfo()

    @abstractmethod
    def __process_features__(self, processed_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        pass


class ScalePreprocessor(MetadataPreprocessor):
    @property
    def has_suffix(self) -> bool:
        return True

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "target",
            to_scale: list[str] | None = None,
            perf_type: str = "abs",  # or "rel"
            remove_outliers: bool = False,
            outlier_modifier: float = 1.0,
            test_mode: bool = False,
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            to_scale=to_scale,
            test_mode=test_mode,
        )
        self.parameters = {}
        self.perf_type = perf_type
        self.remove_outliers = remove_outliers
        self.outlier_modifier = outlier_modifier

    def __process_features__(
            self,
            processed_dataset: pd.DataFrame
    ) -> tuple[pd.DataFrame, HandlerInfo]:
        if self.remove_outliers:
            q1 = processed_dataset.quantile(0.25, axis="index")
            q3 = processed_dataset.quantile(0.75, axis="index")
            iqr = q3 - q1

            lower = q1 - self.outlier_modifier * iqr
            upper = q3 + self.outlier_modifier * iqr

            for i, feature in enumerate(processed_dataset.columns):
                feature_col = processed_dataset[feature]
                feature_col[feature_col < lower[i]] = lower[i]
                feature_col[feature_col > upper[i]] = upper[i]
                processed_dataset[feature] = feature_col

        scaled_values = processed_dataset.to_numpy(copy=True)
        suffix = []
        for scaler_name in self.to_scale:
            scaled_values = self.scalers[scaler_name]().fit_transform(X=scaled_values)
            suffix.append(scaler_name)
        suffix = None if len(suffix) == 0 else "_".join(suffix)

        res = pd.DataFrame(
            scaled_values,
            columns=processed_dataset.columns,
            index=processed_dataset.index
        )
        remove_constant_features(res)

        handler_info = HandlerInfo(suffix=suffix)

        return res, handler_info


class CorrelationPreprocessor(MetadataPreprocessor):
    @property
    def has_suffix(self) -> bool:
        return False

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            to_scale: list[str] | None = None,
            corr_method: str = "spearman",
            corr_value_threshold: float = 0.9,
            vif_value_threshold: float | None = None,
            vif_count_threshold: float | None = None,
            test_mode: bool = False,
    ):
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            to_scale=to_scale,
            test_mode=test_mode,
        )
        self.corr_method = corr_method
        self.corr_value_threshold = corr_value_threshold
        self.vif_value_threshold = vif_value_threshold
        self.vif_count_threshold = vif_count_threshold

    def __process_features__(
            self,
            processed_dataset: pd.DataFrame
    ) -> tuple[pd.DataFrame, HandlerInfo]:
        corr = processed_dataset.corr(method=self.corr_method)
        collinear_pairs = set()

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                if i != j and (abs(corr.iloc[i, j])) >= self.corr_value_threshold:
                    collinear_pairs.add(tuple(sorted([corr.index[i], corr.columns[j]])))

        corr.drop(set([i[0] for i in collinear_pairs]), inplace=True, axis="index")
        corr.drop(set([i[0] for i in collinear_pairs]), inplace=True, axis="columns")

        if self.vif_count_threshold is None and self.vif_value_threshold is None:
            return processed_dataset.loc[:, corr.index], HandlerInfo()

        sorted_vif = self.compute_vif(processed_dataset.loc[:, corr.columns])
        max_iter = self.vif_count_threshold \
            if self.vif_count_threshold is not None \
            else len(sorted_vif.index)

        for i in range(max_iter):
            vif_max = sorted_vif.max()["VIF"]
            if self.vif_value_threshold is not None and vif_max < self.vif_value_threshold:
                break
            sorted_vif = self.compute_vif(processed_dataset.loc[:, sorted_vif.index[1:]])

        return processed_dataset.loc[:, sorted_vif.index], HandlerInfo()


    @staticmethod
    def compute_vif(dataset: pd.DataFrame) -> pd.DataFrame:
        vif_data = pd.DataFrame(index=dataset.columns)
        vif_data["VIF"] = [variance_inflation_factor(dataset.values, i)
                           for i in range(len(dataset.columns))]
        return vif_data.sort_values(by="VIF", ascending=False)


if __name__ == "__main__":
    corr_filter = CorrelationPreprocessor(
        md_source=TabzillaSource(),
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        corr_method="spearman",
        corr_value_threshold=0.9,
        vif_value_threshold=20000,
        vif_count_threshold=None,
        test_mode=False,
    )

    corr_features, corr_metrics = corr_filter.preprocess(
        feature_suffix="power",
        metrics_suffix="perf_abs"
    )
    print(corr_features.shape)
    print(corr_metrics.shape)
