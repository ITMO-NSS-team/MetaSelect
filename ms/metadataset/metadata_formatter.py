from abc import ABC

import pandas as pd

from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_handler import FeaturesHandler, MetricsHandler
from ms.handler.metadata_source import TabzillaSource


class MetadataFormatter(FeaturesHandler, MetricsHandler, ABC):
    """
    A class for formatting metadata related to features and metrics.

    This class provides methods to retrieve the class name, formatted folder path,
    and resource path from the configuration. It is designed to facilitate the 
    organization and access of metadata in a structured manner.

    Methods:
        class_name: Returns the name of the class.
        class_folder: Retrieves the formatted folder path from the configuration.
        save_path: Retrieves the path to the resources.
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
    which is useful for identifying the type of the object.

    Returns:
        str: The name of the class, which is "formatter".
    """
        return "formatter"

    @property
    def class_folder(self) -> str:
        """
Retrieve the formatted folder path from the configuration.

    This method accesses the configuration object associated with the
    instance and returns the formatted folder path as a string.

    Returns:
        str: The formatted folder path from the configuration.
    """
        return self.config.formatted_folder

    @property
    def save_path(self) -> str:
        """
Retrieve the path to the resources.

    This method returns the path to the resources defined in the configuration.

    Returns:
        str: The path to the resources.
    """
        return self.config.resources

    def __init__(
            self,
            features_folder: str = "raw",
            metrics_folder: str | None = "raw",
            test_mode: bool = False,
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the instance,
    including the folders for features and metrics, as well as the mode
    for testing.

    Args:
        features_folder (str): The path to the folder containing features.
            Defaults to "raw".
        metrics_folder (str | None): The path to the folder containing metrics.
            Defaults to "raw". Can be None if no metrics folder is specified.
        test_mode (bool): A flag indicating whether the instance is in test mode.
            Defaults to False.

    Returns:
        None: This method does not return a value.
    """
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )


class TabzillaFormatter(MetadataFormatter):
    """
    A class for formatting and processing Tabzilla features and metrics datasets.

    This class provides methods to initialize, process, and aggregate features and metrics 
    datasets related to Tabzilla functionality. It includes capabilities for handling 
    datasets, applying transformations, and returning processed results.

    Methods:
        source: Creates and returns a new instance of TabzillaSource.
        has_index: Check the presence of features and metrics indices.
        __init__: Initializes an instance of the class.
        __handle_features__: Processes the given features dataset by aggregating, rounding, and filtering.
        __handle_metrics__: Processes the provided metrics dataset and returns aggregated metrics along with handler information.
        __aggregate_features__: Aggregate features from the given dataset.
        __round_attributes__: Rounds specified attributes in the given features dataset.
        __filter_families__: Filter out specific family features from the dataset.
        __aggregate_metrics__: Aggregate metrics from the provided dataset.
    """
    @property
    def source(self) -> TabzillaSource:
        """
Creates and returns a new instance of TabzillaSource.

    This method initializes a new TabzillaSource object, which can be used 
    for further operations related to Tabzilla functionality.

    Returns:
        TabzillaSource: A new instance of the TabzillaSource class.
    """
        return TabzillaSource()

    @property
    def has_index(self) -> dict:
        """
Check the presence of features and metrics indices.

    This method returns a dictionary indicating whether features and metrics indices are present.

    Returns:
        dict: A dictionary with two keys, 'features' and 'metrics', both set to False.
    """
        return {
            "features": False,
            "metrics": False,
        }

    def __init__(
            self,
            features_folder: str = "raw",
            metrics_folder: str | None = "raw",
            test_mode: bool = False,
            agg_func_features: str = "median",
            agg_func_metrics: str = "mean",
            round_attrs: list[str] | None = None,
            filter_families: list[str] | None = None,
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including
    the folders for features and metrics, the mode of operation, and various
    aggregation functions and attributes.

    Args:
        features_folder (str): The folder path for features. Defaults to "raw".
        metrics_folder (str | None): The folder path for metrics. Defaults to "raw".
        test_mode (bool): A flag indicating whether the instance is in test mode. Defaults to False.
        agg_func_features (str): The aggregation function to use for features. Defaults to "median".
        agg_func_metrics (str): The aggregation function to use for metrics. Defaults to "mean".
        round_attrs (list[str] | None): A list of attributes to round. Defaults to a predefined list if None.
        filter_families (list[str] | None): A list of families to filter. Defaults to None.

    Returns:
        None: This method does not return a value.
    """
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.agg_func_features = agg_func_features
        self.agg_func_metrics = agg_func_metrics
        self.round_attrs = round_attrs if round_attrs is not None else \
            [
                "f__pymfe.general.nr_inst",
                "f__pymfe.general.nr_attr",
                "f__pymfe.general.nr_bin",
                "f__pymfe.general.nr_cat",
                "f__pymfe.general.nr_num",
                "f__pymfe.general.nr_class",
            ]
        self.filter_families = filter_families

    def __handle_features__(self, features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        agg_features = self.__aggregate_features__(features_dataset=features_dataset)
        self.__round_attributes__(features_dataset=agg_features)
        self.__filter_families__(features_dataset=agg_features)
        return agg_features, HandlerInfo()

    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        agg_metrics = self.__aggregate_metrics__(metrics_dataset=metrics_dataset)
        return agg_metrics, HandlerInfo()

    def __aggregate_features__(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        agg_features = features_dataset.groupby("dataset_name")
        if self.agg_func_features == "median":
            agg_features = agg_features.median(numeric_only=True)
        else:
            agg_features = agg_features.mean(numeric_only=True)
        return agg_features

    def __round_attributes__(self, features_dataset: pd.DataFrame) -> None:
        """
Rounds specified attributes in the given features dataset.

    This method checks if there are any attributes to round. If the 
    `round_attrs` attribute is not None, it rounds the values of 
    the specified attributes in the provided DataFrame to the nearest 
    integer.

    Args:
        features_dataset (pd.DataFrame): The DataFrame containing the 
            features whose attributes are to be rounded.

    Returns:
        None: This method modifies the input DataFrame in place and 
            does not return any value.
    """
        if self.round_attrs is None:
            return
        for attr in self.round_attrs:
            if attr in features_dataset.columns:
                features_dataset.loc[:, attr] = features_dataset[attr].round(0)

    def __filter_families__(self, features_dataset: pd.DataFrame) -> None:
        """
Filter out specific family features from the dataset.

    This method removes columns from the provided features dataset that do not
    match the specified family prefixes. If no families are specified for filtering,
    the method will return without making any changes to the dataset.

    Args:
        features_dataset (pd.DataFrame): The dataset containing features to be filtered.
            This DataFrame is expected to have columns that may start with family prefixes.

    Returns:
        None: This method modifies the features_dataset in place and does not return a value.
    """
        if self.filter_families is None:
            return
        prefixes = [f"f__pymfe.{family}" for family in self.filter_families]
        filter_cols = [
            col
            for col in features_dataset.columns
            if not col.startswith("f__")
                or any(col.startswith(prefix) for prefix in prefixes)
        ]
        features_dataset.drop(columns=filter_cols, inplace=True)

    def __aggregate_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        agg_metrics = metrics_dataset.loc[
            metrics_dataset["hparam_source"] == "default"
        ].groupby(
            ["dataset_name", "alg_name"]
        )
        if self.agg_func_metrics == "median":
            agg_metrics = agg_metrics.median(numeric_only=True)
        else:
            agg_metrics = agg_metrics.mean(numeric_only=True)
        return agg_metrics


if __name__ == "__main__":
    formatter = TabzillaFormatter(
        features_folder="raw",
        metrics_folder="raw",
        test_mode=False,
    )
    formatter.handle_features(to_save=True)
    formatter.handle_metrics(to_save=True)
