from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_handler import FeaturesHandler, MetricsHandler
from ms.handler.metadata_source import MetadataSource, TabzillaSource
from ms.utils.metadata import remove_constant_features


class MetadataPreprocessor(FeaturesHandler, MetricsHandler, ABC):
    """
    A class for preprocessing metadata features and metrics datasets.

    This class is responsible for handling the preprocessing of features and metrics datasets,
    including loading, filtering, and processing data based on specified suffixes. It provides
    methods to check for suffixes, retrieve class information, and manage the paths for
    preprocessed data and resources.

    Methods:
        has_suffix: Determines if the current object has a specific suffix.
        class_name: Return the name of the class.
        class_folder: Retrieve the path to the preprocessed folder.
        source: Retrieve the metadata source.
        has_index: Check the availability of features and metrics.
        save_path: Retrieve the path to the resources.
        __init__: Initializes an instance of the class.
        get_common_datasets: Retrieve common datasets based on specified suffixes for features and metrics.
        preprocess: Preprocesses features and metrics datasets.
        __handle_features__: Processes the given features dataset based on common datasets.
        __handle_metrics__: Processes the given metrics dataset and returns processed dataset and handler information.
        __process_features__: Processes the features of the given dataset.
    """
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "power": PowerTransformer,
        "quantile": QuantileTransformer
    }

    @property
    @abstractmethod
    def has_suffix(self) -> bool:
        """
Determines if the current object has a specific suffix.

    This method checks whether the object meets certain criteria 
    to have a suffix. The exact criteria are defined within the 
    method's implementation.

    Returns:
        bool: True if the object has the specified suffix, 
              False otherwise.
    """
        pass

    @property
    def class_name(self) -> str:
        """
Return the name of the class.

    This method returns a string that represents the name of the class 
    to which the instance belongs.

    Returns:
        str: The name of the class, which is "preprocessor".
    """
        return "preprocessor"

    @property
    def class_folder(self) -> str:
        """
Retrieve the path to the preprocessed folder.

    This method accesses the configuration object associated with the instance
    and returns the path to the folder where preprocessed data is stored.

    Returns:
        str: The path to the preprocessed folder.
    """
        return self.config.preprocessed_folder

    @property
    def source(self) -> MetadataSource:
        """
Retrieve the metadata source.

    This method returns the metadata source associated with the current instance.

    Returns:
        MetadataSource: The metadata source object.
    """
        return self._md_source

    @property
    def has_index(self) -> dict:
        """
Check the availability of features and metrics.

    This method returns a dictionary indicating whether features and metrics 
    are available in the current context.

    Returns:
        dict: A dictionary with keys 'features' and 'metrics', both set to 
        True, indicating their availability.
    """
        return {
            "features": True,
            "metrics": True,
        }

    @property
    def save_path(self) -> str:
        """
Retrieve the path to the resources.

    This method returns the path to the resources defined in the 
    configuration of the instance.

    Returns:
        str: The path to the resources.
    """
        return self.config.resources

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "target",
            to_scale: list[str] | None = None,
            test_mode: bool = False,
    ):
        """
Initializes an instance of the class.

    This constructor initializes the object with the provided metadata source, 
    feature and metrics folder paths, a list of features to scale, and a test mode flag.

    Args:
        md_source (MetadataSource): The source of metadata to be used.
        features_folder (str, optional): The folder containing feature data. Defaults to "filtered".
        metrics_folder (str | None, optional): The folder containing metrics data. Defaults to "target".
        to_scale (list[str] | None, optional): A list of features to scale. Defaults to an empty list if None.
        test_mode (bool, optional): A flag indicating whether the object is in test mode. Defaults to False.

    Returns:
        None
    """
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
        """
Retrieve common datasets based on specified suffixes for features and metrics.

    This method loads datasets for features and metrics, filters them based on the provided
    suffixes, and returns a list of datasets that are common to both features and metrics.

    Args:
        feature_suffix (str, optional): The suffix to filter feature datasets. Defaults to None.
        metrics_suffix (str, optional): The suffix to filter metric datasets. Defaults to None.

    Returns:
        list[str]: A list of dataset names that are common to both features and metrics.
    """
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
    """
    A class for preprocessing features by scaling and handling outliers.

    This class is designed to process datasets by removing outliers and scaling features 
    according to specified parameters. It provides methods to check for suffixes and 
    to process the features of a dataset.

    Methods:
        has_suffix: Check if the object has a specific suffix.
        __init__: Initializes an instance of the class.
        __process_features__: Processes the features of the given dataset by removing outliers and scaling.

    Attributes:
        md_source (MetadataSource): The source of metadata to be used.
        features_folder (str): The folder containing feature data.
        metrics_folder (str | None): The folder containing metrics data.
        to_scale (list[str] | None): A list of features to scale.
        perf_type (str): The type of performance measurement.
        remove_outliers (bool): Flag indicating whether to remove outliers.
        outlier_modifier (float): A modifier to apply when handling outliers.
        test_mode (bool): Flag indicating whether to run in test mode.
    """
    @property
    def has_suffix(self) -> bool:
        """
Check if the object has a specific suffix.

    This method determines whether the object meets the criteria for having a suffix.

    Returns:
        bool: True if the object has the specified suffix, False otherwise.
    """
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
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including
    metadata source, folder paths for features and metrics, scaling options,
    performance type, outlier handling, and test mode configuration.

    Args:
        md_source (MetadataSource): The source of metadata to be used.
        features_folder (str, optional): The folder containing feature data. Defaults to "filtered".
        metrics_folder (str | None, optional): The folder containing metrics data. Defaults to "target".
        to_scale (list[str] | None, optional): A list of features to scale. Defaults to None.
        perf_type (str, optional): The type of performance measurement, either "abs" for absolute or "rel" for relative. Defaults to "abs".
        remove_outliers (bool, optional): Flag indicating whether to remove outliers. Defaults to False.
        outlier_modifier (float, optional): A modifier to apply when handling outliers. Defaults to 1.0.
        test_mode (bool, optional): Flag indicating whether to run in test mode. Defaults to False.

    Returns:
        None
    """
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
    """
    A class for preprocessing features in a dataset to reduce multicollinearity 
    by removing collinear variables based on correlation and Variance Inflation Factor (VIF).

    This class provides methods to initialize the preprocessor, process features 
    to eliminate collinearity, compute VIF values, and check for specific suffixes.

    Methods:
        __init__(md_source, features_folder='preprocessed', metrics_folder='preprocessed', 
                 to_scale=None, corr_method='spearman', corr_value_threshold=0.9, 
                 vif_value_threshold=None, vif_count_threshold=None, test_mode=False):
            Initializes an instance of the class with specified parameters.

        has_suffix():
            Check if the object has a specific suffix.

        __process_features__(processed_dataset):
            Processes features in the given dataset to remove collinear variables 
            and reduce multicollinearity based on VIF.

        compute_vif(dataset):
            Compute the Variance Inflation Factor (VIF) for each feature in the dataset.
    """
    @property
    def has_suffix(self) -> bool:
        """
Check if the object has a specific suffix.

    This method determines whether the object meets certain criteria 
    related to suffixes. The specific implementation details may vary 
    based on the context in which this method is used.

    Returns:
        bool: Always returns False in the current implementation.
    """
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
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including
    metadata source, folder paths for features and metrics, scaling options,
    correlation method and thresholds, and VIF (Variance Inflation Factor) 
    thresholds. It also allows for a test mode to be activated.

    Args:
        md_source (MetadataSource): The source of metadata to be used.
        features_folder (str, optional): The folder containing preprocessed features. Defaults to "preprocessed".
        metrics_folder (str | None, optional): The folder containing preprocessed metrics. Defaults to "preprocessed".
        to_scale (list[str] | None, optional): A list of features to scale. Defaults to None.
        corr_method (str, optional): The method used for correlation calculation. Defaults to "spearman".
        corr_value_threshold (float, optional): The threshold for correlation values. Defaults to 0.9.
        vif_value_threshold (float | None, optional): The threshold for VIF values. Defaults to None.
        vif_count_threshold (float | None, optional): The threshold for the count of VIF values. Defaults to None.
        test_mode (bool, optional): Flag to indicate if the instance is in test mode. Defaults to False.

    Returns:
        None
    """
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
