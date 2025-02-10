from abc import ABC

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer

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
        self.common_datasets = self.get_common_datasets(
            feature_suffix=feature_suffix,
            metrics_suffix=metrics_suffix
        )
        features = self.handle_features(
            load_suffix=feature_suffix,
            save_suffix=None,
            to_save=True
        )
        metrics = self.handle_metrics(
            load_suffix=metrics_suffix,
            save_suffix=metrics_suffix,
            to_save=True
        )

        return features, metrics


class ScalePreprocessor(MetadataPreprocessor):
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

    def __handle_features__(self, features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        processed_dataset = features_dataset.copy()
        processed_dataset = processed_dataset.loc[self.common_datasets].sort_index()

        if self.remove_outliers:
            Q1 = processed_dataset.quantile(0.25, axis="index")
            Q3 = processed_dataset.quantile(0.75, axis="index")
            IQR = Q3 - Q1

            lower = Q1 - self.outlier_modifier * IQR
            upper = Q3 + self.outlier_modifier * IQR

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

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        new_metrics_dataset = metrics_dataset.loc[self.common_datasets].sort_index()
        return new_metrics_dataset, HandlerInfo()


if __name__ == "__main__":
    preprocessor = ScalePreprocessor(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="target",
        to_scale=["power"],
        perf_type="abs",
        remove_outliers=False,
        outlier_modifier=1.5,
        test_mode=False,
    )
    features, metrics = preprocessor.preprocess(
        feature_suffix=None,
        metrics_suffix="perf_abs"
    )
    print(features.shape)
    print(metrics.shape)
