from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_handler import MetricsHandler
from ms.handler.metadata_source import MetadataSource, TabzillaSource
from ms.utils.typing import NDArrayFloatT


class TargetBuilder(MetricsHandler, ABC):
    """
    A class to build and manage target datasets for machine learning models.

    This class provides methods to retrieve metadata sources, check the availability of features and metrics,
    and manage paths and configurations related to target datasets. It also includes functionality for processing
    and rearranging metrics datasets for analysis.

    Methods:
        __init__(md_source, features_folder="filtered", metrics_folder="filtered", test_mode=False, 
                 metric_name="F1__test", index_name="dataset_name", alg_name="alg_name"):
            Initializes an instance of the class.

        source():
            Retrieve the metadata source.

        has_index():
            Check the availability of features and metrics.

        save_path():
            Retrieve the path to the resources.

        class_folder():
            Retrieve the target folder from the configuration.

        __handle_metrics__(metrics_dataset):
            Processes the given metrics dataset and returns a rearranged DataFrame along with handler information.

        __rearrange_dataset__(metrics_dataset):
            Rearranges the given metrics dataset into a pivot table format.

        __get_target__(metrics_dataset):
            Extracts the target variable from the provided metrics dataset.

        __get_col_names__(metrics_dataset):
            Retrieve column names from the given metrics dataset.

        __get_suffix__():
            Retrieve the suffix associated with the instance.
    """
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

    This method returns a dictionary indicating whether features and metrics are available.

    Returns:
        dict: A dictionary with keys 'features' and 'metrics', both set to True.
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

    @property
    def class_folder(self) -> str:
        """
Retrieve the target folder from the configuration.

    This method accesses the configuration object associated with the instance
    and returns the value of the target folder.

    Returns:
        str: The target folder specified in the configuration.
    """
        return self.config.target_folder

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
    ):
        """
Initializes an instance of the class.

    This constructor initializes the object with the provided metadata source,
    feature and metrics folder paths, and various configuration options for 
    testing and metrics.

    Args:
        md_source (MetadataSource): The source of metadata to be used.
        features_folder (str, optional): The folder containing feature data. 
            Defaults to "filtered".
        metrics_folder (str | None, optional): The folder containing metrics data. 
            Defaults to "filtered".
        test_mode (bool, optional): Flag indicating whether to run in test mode. 
            Defaults to False.
        metric_name (str, optional): The name of the metric to be used. 
            Defaults to "F1__test".
        index_name (str, optional): The name of the index to be used. 
            Defaults to "dataset_name".
        alg_name (str, optional): The name of the algorithm to be used. 
            Defaults to "alg_name".

    Returns:
        None
    """
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.metric_name = metric_name
        self.index_name = index_name
        self.alg_name = alg_name


    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        metric_results = self.__rearrange_dataset__(
            metrics_dataset=metrics_dataset
        )
        target_array = self.__get_target__(
            metrics_dataset=metric_results
        )
        target_cols = self.__get_col_names__(
            metrics_dataset=metric_results
        )
        handler_info = HandlerInfo(suffix=self.__get_suffix__())
        return (pd.DataFrame(target_array, columns=target_cols, index=metric_results.index),
                handler_info)

    def __rearrange_dataset__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        return metrics_dataset.pivot_table(
            values=self.metric_name,
            index=self.index_name,
            columns=self.alg_name,
            aggfunc='first'
        )

    @abstractmethod
    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        """
Retrieve column names from the given metrics dataset.

    This method extracts and returns the names of the columns from the 
    provided metrics dataset.

    Args:
        metrics_dataset (pd.DataFrame): The DataFrame containing the metrics 
        from which to extract column names.

    Returns:
        list[str]: A list of column names from the metrics dataset.
    """
        pass

    @abstractmethod
    def __get_suffix__(self) -> str:
        """
Retrieve the suffix associated with the instance.

    This method is intended to return a string that represents the 
    suffix related to the current instance of the class.

    Returns:
        str: The suffix associated with the instance.
    """
        pass


class TargetRawBuilder(TargetBuilder):
    """
    A class for building target values from a metrics dataset.

    This class is responsible for initializing parameters related to metadata, 
    feature and metric folders, and configuration options. It provides methods 
    to extract target values and column names from a given metrics dataset, 
    as well as to retrieve the class name and its suffix.

    Methods:
        class_name: Returns the name of the class.
        __init__: Initializes an instance of the class with specified parameters.
        __get_target__: Extracts the target values from the provided metrics dataset.
        __get_col_names__: Retrieves the column names from the provided metrics dataset.
        __get_suffix__: Retrieves the suffix of the class name.
    """
    @property
    def class_name(self) -> str:
        """
Return the name of the class.

    This method returns a string that represents the name of the class 
    to which the instance belongs.

    Returns:
        str: The name of the class, which is always "raw".
    """
        return "raw"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including
    metadata source, folder paths for features and metrics, and configuration
    options for testing and metrics.

    Args:
        md_source (MetadataSource): The source of metadata to be used.
        features_folder (str, optional): The folder containing feature data. Defaults to "filtered".
        metrics_folder (str | None, optional): The folder containing metric data. Defaults to "filtered".
        test_mode (bool, optional): Flag indicating whether to run in test mode. Defaults to False.
        metric_name (str, optional): The name of the metric to be used. Defaults to "F1__test".
        index_name (str, optional): The name of the dataset index. Defaults to "dataset_name".
        alg_name (str, optional): The name of the algorithm. Defaults to "alg_name".

    Returns:
        None: This method does not return a value.
    """
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )

    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        target_raw = metrics_dataset.to_numpy()
        return target_raw

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        """
Retrieve the column names from the provided metrics dataset.

    This method extracts and returns the names of the columns from the 
    given pandas DataFrame.

    Args:
        metrics_dataset (pd.DataFrame): The DataFrame from which to 
            retrieve the column names.

    Returns:
        list[str]: A list containing the names of the columns in the 
            metrics dataset.
    """
        return metrics_dataset.columns

    def __get_suffix__(self) -> str:
        """
Retrieve the suffix of the class name.

    This method returns the class name of the instance, which can be used 
    as a suffix for various purposes, such as naming conventions or 
    categorization.

    Returns:
        str: The class name of the instance.
    """
        return self.class_name


class TargetPerfBuilder(TargetBuilder):
    """
    A class to build and evaluate target performance metrics from a given dataset.

    This class is responsible for initializing parameters related to performance evaluation,
    processing performance metrics, and generating relevant column names and suffixes for 
    performance data. It supports both absolute and relative performance measurements.

    Methods:
        __init__: Initializes an instance of the class with specified parameters.
        class_name: Returns the name of the class.
        __get_target__: Retrieves and processes the target performance metrics from the dataset.
        __get_col_names__: Generates column names based on the provided metrics dataset.
        __get_suffix__: Generates a suffix based on the class name and performance type.
        __get_abs_perf__: Transforms the input array using discretization.
        __get_rel_perf__: Calculates relative performance rankings for each row in the input array.
    """
    @property
    def class_name(self) -> str:
        """
Return the name of the class.

    This method returns a string that represents the name of the class.

    Returns:
        str: The name of the class, which is "perf".
    """
        return "perf"

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            perf_type: str = "abs", # or "rel"
            n_bins: int = 2,
            strategy: str = "quantile",
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including
    metadata source, folder paths for features and metrics, and various configuration
    options for performance evaluation.

    Args:
        md_source (MetadataSource): The source of metadata to be used.
        features_folder (str, optional): The folder containing feature data. Defaults to "filtered".
        metrics_folder (str | None, optional): The folder containing metrics data. Defaults to "filtered".
        test_mode (bool, optional): Flag indicating whether to run in test mode. Defaults to False.
        metric_name (str, optional): The name of the metric to evaluate. Defaults to "F1__test".
        index_name (str, optional): The name of the dataset index. Defaults to "dataset_name".
        alg_name (str, optional): The name of the algorithm being used. Defaults to "alg_name".
        perf_type (str, optional): The type of performance measurement, either "abs" or "rel". Defaults to "abs".
        n_bins (int, optional): The number of bins to use for performance evaluation. Defaults to 2.
        strategy (str, optional): The strategy for binning, e.g., "quantile". Defaults to "quantile".

    Returns:
        None
    """
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )
        self.perf_type = perf_type
        self.n_bins = n_bins
        self.strategy = strategy

    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        target_perf = metrics_dataset.to_numpy(copy=True)
        target_perf = np.where(np.isnan(target_perf), -np.inf, target_perf)

        if self.perf_type == "abs":
            target_perf = self.__get_abs_perf__(nd_array=target_perf)
        elif self.perf_type == "rel":
            target_perf = self.__get_rel_perf__(nd_array=target_perf)
        else:
            raise ValueError(f"Unsupported performance metric: {self.perf_type}")

        return target_perf

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        """
Generate column names based on the provided metrics dataset.

    This method constructs a list of column names by appending a performance type suffix 
    to each algorithm name found in the given metrics dataset.

    Args:
        metrics_dataset (pd.DataFrame): A DataFrame containing metrics data with algorithm names as columns.

    Returns:
        list[str]: A list of formatted column names, each consisting of the algorithm name 
                    followed by a performance type suffix.
    """
        cols = []
        for alg_name in metrics_dataset.columns:
            cols.append(f"{alg_name}__{self.perf_type}perf")
        return cols

    def __get_suffix__(self) -> str:
        """
Generate a suffix based on the class name and performance type.

    This method constructs a string suffix by combining the class name
    and the performance type, separated by an underscore.

    Returns:
        str: A string representing the suffix in the format
        "{class_name}_{perf_type}".
    """
        return f"{self.class_name}_{self.perf_type}"

    def __get_abs_perf__(self, nd_array: NDArrayFloatT) -> NDArrayFloatT:
        """
Transforms the input array using discretization.

    This method applies a discretization process to each column of the 
    provided NumPy array using the KBinsDiscretizer. It creates a new 
    array where each column is transformed into discrete bins based on 
    the specified number of bins and strategy.

    Args:
        nd_array (NDArrayFloatT): A NumPy array of floats to be transformed. 
                                   The array should have a shape of (n_samples, n_features).

    Returns:
        NDArrayFloatT: A new NumPy array of the same shape as `nd_array`, 
                        where each column has been discretized into bins.
    """
        new_array = np.zeros_like(nd_array)
        disc = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.strategy,
        )
        for i in range(nd_array.shape[1]):
            new_array[:, i] = disc.fit_transform(nd_array[:, i].reshape(-1, 1)).flatten()
        return new_array

    def __get_rel_perf__(self, nd_array: NDArrayFloatT) -> NDArrayFloatT:
        """
Calculate relative performance rankings for each row in the input array.

    This method takes a 2D NumPy array and computes the relative performance 
    rankings for each row. The rankings are determined by sorting the values 
    in descending order, where the highest value receives the rank of 1, 
    the second highest receives a rank of 2, and so on. The resulting 
    rankings are then discretized into bins using the KBinsDiscretizer.

    Args:
        nd_array (NDArrayFloatT): A 2D NumPy array of shape (n_samples, n_features) 
                                   containing the performance values to be ranked.

    Returns:
        NDArrayFloatT: A 2D NumPy array of the same shape as `nd_array`, where each 
                       element represents the discretized relative performance ranking 
                       of the corresponding element in the input array.
    """
        new_array = np.zeros_like(nd_array)
        for i in range(nd_array.shape[0]):
            row = np.argsort(nd_array[i])[::-1]
            for j in range(nd_array.shape[1]):
                new_array[i, row[j]] = j + 1
        disc = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy=self.strategy,
        )
        new_array = disc.fit_transform(new_array.T).T
        return new_array


class TargetDiffBuilder(TargetBuilder):
    """
    A class to build target arrays based on the differences between selected model metrics.

    This class is designed to facilitate the comparison of model performance metrics by generating 
    binary target arrays that indicate which model performs better for each class. It utilizes 
    metadata sources and allows for configuration of various parameters related to features and metrics.

    Methods:
        class_name: Returns the name of the class.
        __init__: Initializes an instance of the class with necessary parameters.
        __get_target__: Generates a target array based on the differences between selected model metrics.
        __get_col_names__: Retrieves column names from the metrics dataset.
        __get_suffix__: Retrieves the suffix of the class name.

    Attributes:
        md_source (MetadataSource): The source of metadata for the model.
        classes (list[str]): A list of class labels for the classification task.
        model_classes (dict[str, str]): A dictionary mapping model names to their respective classes.
        features_folder (str): The folder where features are stored.
        metrics_folder (str | None): The folder where metrics are stored.
        test_mode (bool): Flag indicating whether to run in test mode.
        metric_name (str): The name of the metric to evaluate.
        index_name (str): The name of the index for the dataset.
        alg_name (str): The name of the algorithm being used.
        n_bins (int): The number of bins to use for discretization.
        strategy (str): The strategy for binning.
    """
    @property
    def class_name(self) -> str:
        """
Return the name of the class.

    This method returns a string that represents the name of the class.

    Returns:
        str: The name of the class, which is "diff".
    """
        return "diff"

    def __init__(
            self,
            md_source: MetadataSource,
            classes: list[str],
            model_classes: dict[str, str],
            features_folder: str = "filtered",
            metrics_folder: str | None = "filtered",
            test_mode: bool = False,
            metric_name: str = "F1__test",
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            n_bins: int = 3,
            strategy: str = "quantile",
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including metadata source,
    class labels, model class mappings, and configuration for feature and metric folders.

    Args:
        md_source (MetadataSource): The source of metadata for the model.
        classes (list[str]): A list of class labels for the classification task.
        model_classes (dict[str, str]): A dictionary mapping model names to their respective classes.
        features_folder (str, optional): The folder where features are stored. Defaults to "filtered".
        metrics_folder (str | None, optional): The folder where metrics are stored. Defaults to "filtered".
        test_mode (bool, optional): Flag indicating whether to run in test mode. Defaults to False.
        metric_name (str, optional): The name of the metric to evaluate. Defaults to "F1__test".
        index_name (str, optional): The name of the index for the dataset. Defaults to "dataset_name".
        alg_name (str, optional): The name of the algorithm being used. Defaults to "alg_name".
        n_bins (int, optional): The number of bins to use for discretization. Defaults to 3.
        strategy (str, optional): The strategy for binning. Defaults to "quantile".

    Returns:
        None
    """
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
            metric_name=metric_name,
            index_name=index_name,
            alg_name=alg_name,
        )
        self.classes = classes
        self.model_classes = model_classes
        self.n_bins = n_bins
        self.strategy = strategy
        self._col_name = ""

    def __get_target__(self, metrics_dataset: pd.DataFrame) -> np.ndarray:
        mean_vals = metrics_dataset.mean()
        max_res = {c : ("", 0.) for c in self.classes}
        for i in mean_vals.index:
            if mean_vals[i] > max_res[self.model_classes[i]][1]:
                max_res[self.model_classes[i]] = (i, mean_vals[i])
        models = [max_res[key][0] for key in max_res]

        diff_df = pd.DataFrame(index=metrics_dataset.index)
        res = metrics_dataset[models[0]] - metrics_dataset[models[1]]
        diff_df[f"diff__{models[0]}__{models[1]}"] \
            = [0 if r > 0 else 1 for r in res]

        self._col_name = diff_df.columns[0]

        return diff_df.to_numpy(copy=True)

    def __get_col_names__(self, metrics_dataset: pd.DataFrame) -> list[str]:
        """
Retrieve column names from the metrics dataset.

    This method returns a list containing the column name associated with the
    instance of the class.

    Args:
        metrics_dataset (pd.DataFrame): The dataset from which to retrieve column names.

    Returns:
        list[str]: A list containing the column name.
    """
        return [self._col_name]

    def __get_suffix__(self) -> str:
        """
Retrieve the suffix of the class name.

    This method returns the class name of the instance, which can be used 
    as a suffix for various purposes, such as naming conventions or 
    categorization.

    Returns:
        str: The class name of the instance.
    """
        return self.class_name


if __name__ == "__main__":
    raw_builder = TargetRawBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        test_mode=False,
    )

    abs_perf_builder = TargetPerfBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        perf_type="abs",
        n_bins=2,
        strategy="quantile",
        test_mode=False,
    )

    rel_perf_builder = TargetPerfBuilder(
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        perf_type="rel",
        n_bins=3,
        strategy="uniform",
        test_mode=False,
    )

    model_classes = {
        "rtdl_FTTransformer": "nn",
        "rtdl_MLP": "nn",
        "rtdl_ResNet": "nn",
        "LinearModel": "classic",
        "RandomForest": "classic",
        "XGBoost": "classic"
    }

    diff_builder = TargetDiffBuilder(
        classes=["nn", "classic"],
        model_classes=model_classes,
        md_source=TabzillaSource(),
        features_folder="filtered",
        metrics_folder="filtered",
        metric_name="F1__test",
        n_bins=3,
        strategy="uniform",
        test_mode=False,
    )

    raw = raw_builder.handle_metrics()
    abs = abs_perf_builder.handle_metrics()
    rel = rel_perf = rel_perf_builder.handle_metrics()
    diff = diff_builder.handle_metrics()
