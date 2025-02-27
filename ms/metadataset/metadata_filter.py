from abc import ABC
from statistics import median

import numpy as np
import pandas as pd

from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_handler import FeaturesHandler, MetricsHandler
from ms.handler.metadata_source import TabzillaSource
from ms.utils.metadata import remove_constant_features


class MetadataFilter(FeaturesHandler, MetricsHandler, ABC):
    """
    A class to filter and manage metadata related to features and metrics.

    This class provides methods to retrieve class information, check the 
    availability of features and metrics, and access resource paths defined 
    in the configuration.

    Methods:
        class_name: Returns the name of the class.
        has_index: Checks the availability of features and metrics.
        save_path: Retrieves the path to the resources.
        class_folder: Retrieves the path of the filtered folder from the configuration.
        __init__: Initializes an instance of the class.

    Attributes:
        features_folder (str): The path to the folder containing features.
        metrics_folder (str | None): The path to the folder containing metrics.
        test_mode (bool): A flag indicating whether the instance is in test mode.
    """
    @property
    def class_name(self) -> str:
        """
Return the name of the class.

    This method returns a string that represents the name of the class, 
    which is "filter".

    Returns:
        str: The name of the class.
    """
        return "filter"

    @property
    def has_index(self) -> dict:
        """
Check the availability of features and metrics.

    This method returns a dictionary indicating whether certain features
    and metrics are available.

    Returns:
        dict: A dictionary with the following keys:
            - "features" (bool): Indicates if features are available.
            - "metrics" (bool): Indicates if metrics are available.
    """
        return {
            "features": True,
            "metrics": False,
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

    @property
    def class_folder(self) -> str:
        """
Retrieve the path of the filtered folder from the configuration.

    This method accesses the configuration object associated with the instance
    and returns the path of the filtered folder as a string.

    Returns:
        str: The path of the filtered folder.
    """
        return self.config.filtered_folder

    def __init__(
            self,
            features_folder: str = "formatted",
            metrics_folder: str | None = "formatted",
            test_mode: bool = False,
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the instance,
    including the paths for features and metrics folders, as well as 
    the mode of operation (test mode).

    Args:
        features_folder (str): The path to the folder containing features.
            Defaults to "formatted".
        metrics_folder (str | None): The path to the folder containing metrics.
            Defaults to "formatted". Can be None if no metrics folder is specified.
        test_mode (bool): A flag indicating whether the instance is in test mode.
            Defaults to False.

    Returns:
        None
    """
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )


class TabzillaFilter(MetadataFilter):
    """
    A class for processing and filtering datasets of features and metrics.

    This class provides methods to clean, filter, and refine datasets containing features and metrics,
    ensuring that the data meets specified criteria for analysis. It includes functionality for handling
    missing values, removing unsuitable features, and filtering datasets based on exclusion criteria.

    Methods:
        source: Create and return a new instance of TabzillaSource.
        __init__: Initializes an instance of the class.
        __handle_features__: Processes and filters the given features dataset.
        __handle_metrics__: Processes the given metrics dataset by applying various filters.
        __remove_unsuitable_features__: Remove unsuitable features from the given dataset.
        __fill_undefined_values__: Fill undefined values in the given features dataset.
        __remove_duplicates__: Remove duplicate rows from the features dataset.
        __remove_features_by_func__: Remove features from the dataset based on specified exclusion criteria.
        __remove_datasets_by_name__: Remove specified datasets from the provided DataFrame.
        __remove_features_by_key__: Remove features from the dataset based on specified exclusion keys.
        __filter_outliers__: Filters outliers from the given features dataset based on a predefined threshold.
        __filter_models__: Filters the models in the given metrics dataset based on a predefined list of models.
        __filter_datasets_by_model__: Filters the provided metrics dataset to retain only those datasets that match the specified models.

    Attributes:
        features_folder (str): The folder path where feature data is stored.
        metrics_folder (str | None): The folder path where metric data is stored.
        test_mode (bool): A flag indicating whether the instance is in test mode.
        nan_threshold (float): The threshold for the proportion of NaN values allowed in the data.
        fill_func (str): The method used to fill missing values.
        funcs_to_exclude (list[str] | None): A list of functions to exclude from processing.
        keys_to_exclude (list[str] | None): A list of keys to exclude from the dataset.
        datasets_to_exclude (list[str] | None): A list of datasets to exclude from processing.
        models_list (list[str] | None): A list of models to include in processing.
        value_threshold (float): The threshold for filtering values in the dataset.
    """
    @property
    def source(self) -> TabzillaSource:
        """
Create and return a new instance of TabzillaSource.

    This method initializes a new TabzillaSource object and returns it.

    Returns:
        TabzillaSource: A new instance of the TabzillaSource class.
    """
        return TabzillaSource()

    def __init__(
            self,
            features_folder: str = "formatted",
            metrics_folder: str | None = "formatted",
            test_mode: bool = False,
            nan_threshold: float = 0.5,
            fill_func: str = "median",
            funcs_to_exclude: list[str] | None = None,
            keys_to_exclude: list[str] | None = None,
            datasets_to_exclude: list[str] | None = None,
            models_list: list[str] | None = None,
            value_threshold: float = 10e6,
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including
    paths for features and metrics, configuration for test mode, and various
    thresholds and exclusion lists for data processing.

    Args:
        features_folder (str): The folder path where feature data is stored.
            Defaults to "formatted".
        metrics_folder (str | None): The folder path where metric data is stored.
            Defaults to "formatted". Can be None.
        test_mode (bool): A flag indicating whether the instance is in test mode.
            Defaults to False.
        nan_threshold (float): The threshold for the proportion of NaN values
            allowed in the data. Defaults to 0.5.
        fill_func (str): The method used to fill missing values. Defaults to "median".
        funcs_to_exclude (list[str] | None): A list of functions to exclude from processing.
            Can be None.
        keys_to_exclude (list[str] | None): A list of keys to exclude from the dataset.
            Can be None.
        datasets_to_exclude (list[str] | None): A list of datasets to exclude from processing.
            Can be None.
        models_list (list[str] | None): A list of models to include in processing.
            Can be None.
        value_threshold (float): The threshold for filtering values in the dataset.
            Defaults to 10e6.

    Returns:
        None
    """
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.nan_threshold = nan_threshold
        self.fill_func = fill_func
        self.funcs_to_exclude = funcs_to_exclude
        self.keys_to_exclude = keys_to_exclude
        self.datasets_to_exclude = datasets_to_exclude
        self.models_list = models_list
        self.value_threshold = value_threshold


    def __handle_features__(self, features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        filtered_features = features_dataset.copy()

        self.__remove_features_by_func__(features_dataset=filtered_features)
        self.__remove_datasets_by_name__(dataset=filtered_features)
        self.__remove_features_by_key__(features_dataset=filtered_features)
        self.__remove_unsuitable_features__(features_dataset=filtered_features)
        self.__filter_outliers__(features_dataset=filtered_features)
        self.__fill_undefined_values__(features_dataset=filtered_features)
        remove_constant_features(features_dataset=filtered_features)
        filtered_features = self.__remove_duplicates__(features_dataset=filtered_features)

        return filtered_features, HandlerInfo()

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        filtered_metrics = metrics_dataset.copy()

        self.__remove_datasets_by_name__(dataset=filtered_metrics)
        self.__filter_models__(metrics_dataset=filtered_metrics)
        self.__filter_datasets_by_model__(metrics_dataset=filtered_metrics)

        return filtered_metrics, HandlerInfo()

    def __remove_unsuitable_features__(self, features_dataset: pd.DataFrame) -> None:
        """
Remove unsuitable features from the given dataset.

    This method iterates through the columns of the provided features dataset
    and removes any columns that exceed a specified threshold of missing values
    or contain only constant values.

    Args:
        features_dataset (pd.DataFrame): The dataset containing features to be evaluated.
            Each column in the DataFrame represents a feature, and the method will
            assess each feature based on its missing values and constant values.

    Returns:
        None: This method modifies the input DataFrame in place and does not return a value.
    """
        num_datasets = len(features_dataset.index)
        for col in features_dataset:
            x = features_dataset[col].to_numpy()
            if (features_dataset[col].isna().sum() > num_datasets * self.nan_threshold
                    or np.all(x == x[0])):
                features_dataset.drop(col, axis="columns", inplace=True)

    def __fill_undefined_values__(self, features_dataset: pd.DataFrame) -> None:
        """
Fill undefined values in the given features dataset.

    This method fills missing values in the provided DataFrame using either
    the median or mean of the numeric columns, depending on the specified
    fill function.

    Args:
        features_dataset (pd.DataFrame): The DataFrame containing features
            with potential undefined (NaN) values that need to be filled.

    Returns:
        None: This method modifies the input DataFrame in place and does not
        return a value.
    """
        if self.fill_func == "median":
            values = features_dataset.median(numeric_only=True)
        else:
            values = features_dataset.mean(numeric_only=True)
        features_dataset.fillna(values, inplace=True)

    @staticmethod
    def __remove_duplicates__(features_dataset: pd.DataFrame) -> pd.DataFrame:
        return features_dataset.drop_duplicates().T.drop_duplicates().T

    def __remove_features_by_func__(self, features_dataset: pd.DataFrame) -> None:
        """
Remove features from the dataset based on specified exclusion criteria.

    This method examines the columns of the provided features dataset and removes
    any features that match certain criteria defined by the `funcs_to_exclude` attribute.
    Specifically, it excludes features that have a function name listed in `funcs_to_exclude`
    or those that end with "relative". 

    Args:
        features_dataset (pd.DataFrame): The dataset containing features to be evaluated
            for removal. Each column represents a feature.

    Returns:
        None: This method modifies the input DataFrame in place and does not return a value.
    """
        if self.funcs_to_exclude is not None:
            features_to_remove = []
            for feature in features_dataset.columns:
                f_name = feature.split(".")
                if len(f_name) == 3:
                    continue
                f_func = f_name[3]
                if f_name[-1] == "relative":
                    features_to_remove.append(feature)
                for key in self.funcs_to_exclude:
                    if f_func == key:
                        features_to_remove.append(feature)
            features_dataset.drop(features_to_remove, axis="columns", inplace=True)

    def __remove_datasets_by_name__(self, dataset: pd.DataFrame) -> None:
        """
Remove specified datasets from the provided DataFrame.

    This method removes rows from the given DataFrame that correspond to 
    the datasets listed in the instance's `datasets_to_exclude` attribute. 
    If `datasets_to_exclude` is None, no rows are removed.

    Args:
        dataset (pd.DataFrame): The DataFrame from which to remove datasets.

    Returns:
        None: This method modifies the DataFrame in place and does not return a value.
    """
        if self.datasets_to_exclude is not None:
            dataset.drop(self.datasets_to_exclude, axis="index", inplace=True)

    def __remove_features_by_key__(self, features_dataset: pd.DataFrame) -> None:
        """
Remove features from the dataset based on specified exclusion keys.

    This method checks the columns of the provided features dataset and removes any
    columns that contain keys specified in the `keys_to_exclude` attribute of the
    instance. The removal is done in place, modifying the original DataFrame.

    Args:
        features_dataset (pd.DataFrame): The DataFrame from which features will be removed.
            It should contain the features as columns.

    Returns:
        None: This method does not return a value. It modifies the input DataFrame in place.
    """
        if self.keys_to_exclude is not None:
            features_to_remove = []
            for feature in features_dataset.columns:
                for key in self.keys_to_exclude:
                    if key in feature:
                        features_to_remove.append(feature)
                        break
            features_dataset.drop(features_to_remove, axis="columns", inplace=True)

    def __filter_outliers__(self, features_dataset: pd.DataFrame) -> None:
        """
Filters outliers from the given features dataset based on a predefined threshold.

    This method identifies outliers in each feature of the provided dataset by comparing
    the values against a specified threshold. Features with a number of outliers greater
    than the median outlier count are removed from the dataset, along with the corresponding
    rows that contain outlier values.

    Args:
        features_dataset (pd.DataFrame): A pandas DataFrame containing the features from which
                                          outliers will be filtered. Each column represents a
                                          feature, and each row represents an observation.

    Returns:
        None: This method modifies the input DataFrame in place by dropping features and
              rows that contain outliers.
    """
        outliers_dict = {}
        outliers_list = []

        for i, feature in enumerate(features_dataset.columns):
            feature_outliers = []
            for j, val in enumerate(features_dataset[feature]):
                if val > self.value_threshold or val < -self.value_threshold:
                    feature_outliers.append(j)
            if len(feature_outliers) > 1:
                outliers_dict[feature] = feature_outliers
                outliers_list.append(len(feature_outliers))
            elif len(feature_outliers) == 1:
                outliers_dict[feature] = feature_outliers
            else:
                pass
        median_outlier_count = median(outliers_list)

        features_to_drop = []
        for feature in features_dataset.columns:
            if (outliers_dict.get(feature) is not None
                    and len(outliers_dict[feature]) > median_outlier_count):
                outliers_dict.pop(feature)
                features_to_drop.append(feature)
        features_dataset.drop(features_to_drop, axis="columns", inplace=True)

        datasets_to_drop = set()
        for feature in outliers_dict:
            for dataset_idx in outliers_dict[feature]:
                datasets_to_drop.add(dataset_idx)
        datasets_to_drop = [features_dataset.index[i] for i in datasets_to_drop]
        features_dataset.drop(datasets_to_drop, axis="index", inplace=True)

    def __filter_models__(self, metrics_dataset: pd.DataFrame) -> None:
        """
Filters the models in the given metrics dataset based on a predefined list of models.

    This method iterates through the provided metrics dataset and removes any rows 
    where the algorithm name is not present in the instance's models list. 

    Args:
        metrics_dataset (pd.DataFrame): A pandas DataFrame containing metrics data, 
                                         where each row represents a model's metrics 
                                         and includes an 'alg_name' column.

    Returns:
        None: This method modifies the input DataFrame in place and does not return a value.
    """
        if self.models_list is not None:
            for index, row in metrics_dataset.iterrows():
                if row["alg_name"] not in self.models_list:
                    metrics_dataset.drop(index, axis="index", inplace=True)

    def __filter_datasets_by_model__(self, metrics_dataset: pd.DataFrame):
        """
Filters the provided metrics dataset to retain only those datasets 
    that match the specified models in the instance's models list.

    This method iterates through the given metrics dataset and removes any 
    entries where the dataset's algorithms do not match the models specified 
    in the instance's `models_list`. The filtering is done in place, modifying 
    the original DataFrame.

    Args:
        metrics_dataset (pd.DataFrame): A pandas DataFrame containing metrics 
        data with at least two columns: 'dataset_name' and 'alg_name'. Each 
        row represents a metric associated with a specific dataset and algorithm.

    Returns:
        None: This method modifies the input DataFrame in place and does not 
        return a value.
    """
        if self.models_list is not None:
            dataset_models = {}
            for index, row in metrics_dataset.iterrows():
                if row["dataset_name"] not in dataset_models:
                    dataset_models[row["dataset_name"]] = set()
                dataset_models[row["dataset_name"]].add(row["alg_name"])
            for index, row in metrics_dataset.iterrows():
                if dataset_models[row["dataset_name"]] != set(self.models_list):
                    metrics_dataset.drop(index, axis="index", inplace=True)


if __name__ == "__main__":
    md_filter = TabzillaFilter(
        features_folder="formatted",
        metrics_folder="formatted",
        funcs_to_exclude=[
            "count",
            "histogram",
            "iq_range",
            "median",
            "quantiles",
            "range",
        ],
        models_list=["XGBoost", "RandomForest", "LinearModel",
                     "rtdl_ResNet", "rtdl_FTTransformer", "rtdl_MLP"],
        test_mode=False,
        value_threshold=1e6,
    )

    f = md_filter.handle_features(to_save=True)
    print(f.shape)
    m = md_filter.handle_metrics(to_save=True)
    print(m.shape)
