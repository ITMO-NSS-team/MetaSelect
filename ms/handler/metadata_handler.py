import json
import os
from abc import abstractmethod, ABC
from pathlib import Path

import pandas as pd

from ms.config.metadata_config import MetadataConfig
from ms.handler.handler_info import HandlerInfo
from ms.handler.metadata_source import SourceBased
from ms.utils.debug import Debuggable
from ms.utils.navigation import pjoin


class MetadataHandler(SourceBased, Debuggable, ABC):
    """
    A class to manage metadata configurations and handle data loading and saving operations.

    This class provides methods to configure and retrieve metadata, load and save data 
    in various formats (CSV and JSON), and manage folder structures for features and metrics.

    Methods:
        __init__: Initializes an instance of the class.
        config: Retrieve or set the metadata configuration.
        data_folder: Retrieve or set the data folder information.
        has_index: Check for the presence of indices.
        load_path: Load the resource path from the configuration.
        save_path: Generate and return the path where data should be saved.
        load: Load a CSV file into a pandas DataFrame.
        load_json: Load a JSON file and return its contents as a dictionary.
        save: Saves a DataFrame to a specified CSV file.
        save_json: Saves a dictionary as a JSON file in the specified folder.
        get_path: Constructs a file path based on the provided folder and file names.
        load_features: Load features dataset from a specified folder and file.
        save_features: Saves the processed features to a specified location.
        load_metrics: Load metrics dataset from a specified folder and file.
        save_metrics: Saves the provided metrics to a specified location.
        save_samples: Saves the provided data as a JSON file.
        load_samples: Load sample data from a JSON file.
        get_file_name: Generate a file name based on the provided prefix and optional suffix.
    """
    def __init__(
            self,
            features_folder: str,
            metrics_folder: str | None = None,
            test_mode: bool = False,
    ):
        """
Initializes an instance of the class.

    This constructor sets up the data folder configuration for features and metrics.
    It also initializes the metadata configuration and handles the test mode setting.

    Args:
        features_folder (str): The path to the folder containing feature data.
        metrics_folder (str | None, optional): The path to the folder containing metric data.
            If None, the metrics folder will default to the features folder. Defaults to None.
        test_mode (bool, optional): A flag indicating whether the instance is in test mode.
            Defaults to False.

    Returns:
        None
    """
        super().__init__(
            test_mode=test_mode,
        )
        self._config = MetadataConfig()

        if metrics_folder is None:
            _metrics_folder = features_folder
        else:
            _metrics_folder = metrics_folder
        self._data_folder = {
            "features": features_folder,
            "metrics": _metrics_folder
        }

    @property
    def config(self) -> MetadataConfig:
        """
Retrieve the metadata configuration.

    This method returns the current metadata configuration object associated with the instance.

    Returns:
        MetadataConfig: The metadata configuration object.
    """
        return self._config

    @config.setter
    def config(self, config: MetadataConfig) -> None:
        """
Sets the configuration for the instance.

    This method assigns the provided configuration object to the instance's 
    internal configuration attribute.

    Args:
        config (MetadataConfig): The configuration object to be set.

    Returns:
        None
    """
        self._config = config

    @property
    def data_folder(self) -> dict[str, str]:
        """
Retrieve the data folder information.

    This method returns a dictionary containing the data folder details, 
    where the keys are strings representing the folder names and the values 
    are strings representing the corresponding folder paths.

    Returns:
        dict[str, str]: A dictionary mapping folder names to their paths.
    """
        return self._data_folder

    @data_folder.setter
    def data_folder(self, data_folder: dict[str, str]) -> None:
        """
Sets the data folder for the instance.

    This method assigns the provided data folder dictionary to the 
    instance variable `_data_folder`. The dictionary should contain 
    string keys and string values representing the folder structure.

    Args:
        data_folder (dict[str, str]): A dictionary where keys are 
            folder names and values are their corresponding paths.

    Returns:
        None
    """
        self._data_folder = data_folder

    @property
    @abstractmethod
    def has_index(self) -> dict[str, bool]:
        """
Check for the presence of indices.

    This method checks whether specific indices exist and returns a dictionary
    indicating the presence of each index as a boolean value.

    Returns:
        dict[str, bool]: A dictionary where the keys are index names and the
        values are booleans indicating whether each index exists (True) or not (False).
    """
        pass

    @property
    def load_path(self) -> str:
        """
Load the resource path from the configuration.

    This method retrieves the resource path defined in the configuration object.

    Returns:
        str: The resource path as a string.
    """
        return self.config.resources

    @property
    @abstractmethod
    def save_path(self) -> str:
        """
Generate and return the path where data should be saved.

    This method constructs the appropriate file path for saving data 
    based on the current state of the object. The returned path can 
    be used to store files in a designated directory.

    Returns:
        str: The file path where data will be saved.
    """
        pass

    def load(
            self,
            folder_name: str,
            file_name: str,
            get_index: bool = True,
            inner_folders: list[str] = None,
            path: Path | None = None,
    ) -> pd.DataFrame:
        load_path = self.get_path(
            folder_name=folder_name,
            file_name=file_name,
            inner_folders=inner_folders,
            to_save=False
        ) if path is None else path

        if get_index:
            data_frame = pd.read_csv(load_path, index_col=0)
        else:
            data_frame = pd.read_csv(load_path)

        return data_frame

    def load_json(
            self,
            folder_name: str,
            file_name: str,
            inner_folders: list[str] = None,
            path: Path | None = None,
            to_save: bool = False,
    ) -> dict:
        """
Load a JSON file and return its contents as a dictionary.

    This method constructs the path to the specified JSON file using the provided
    folder name, file name, and optional inner folders. If a path is not provided,
    it will use the constructed path. The JSON file is then read and parsed into a
    Python dictionary.

    Args:
        folder_name (str): The name of the folder containing the JSON file.
        file_name (str): The name of the JSON file to load.
        inner_folders (list[str], optional): A list of inner folder names to include in the path.
            Defaults to None.
        path (Path | None, optional): An optional path to the JSON file. If provided, this path
            will be used instead of constructing one. Defaults to None.
        to_save (bool, optional): A flag indicating whether the file is intended to be saved.
            Defaults to False.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
        load_path = self.get_path(
            folder_name=folder_name,
            file_name=file_name,
            inner_folders=inner_folders,
            to_save=to_save
        ) if path is None else path

        with open(load_path, "r") as f:
            data = json.load(f)
        return data

    def save(
            self,
            data_frame: pd.DataFrame,
            folder_name: str,
            file_name: str,
            inner_folders: list[str] = None,
            save_if_exists: bool = True,
            path: Path | None = None,
    ) -> str:
        """
Saves a DataFrame to a specified CSV file.

    This method saves the provided DataFrame to a CSV file at the specified 
    location. If the file already exists and the `save_if_exists` parameter 
    is set to False, the method will return the existing file path without 
    overwriting it. If the file does not exist, it will create the necessary 
    directories and save the DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame to be saved.
        folder_name (str): The name of the folder where the file will be saved.
        file_name (str): The name of the file to save the DataFrame as.
        inner_folders (list[str], optional): A list of inner folder names to 
            create within the specified folder. Defaults to None.
        save_if_exists (bool, optional): If True, the method will overwrite 
            the existing file. If False, it will not overwrite and will return 
            the existing file path. Defaults to True.
        path (Path | None, optional): An optional path to save the file. If 
            provided, this will override the constructed path from folder_name 
            and file_name. Defaults to None.

    Returns:
        str: The path to the saved CSV file.
    """
        save_path = self.get_path(
            folder_name=folder_name,
            file_name=file_name,
            inner_folders=inner_folders,
        ) if path is None else path

        if os.path.isfile(save_path) and not save_if_exists:
            return save_path

        save_path.parent.mkdir(parents=True, exist_ok=True)
        if (data_frame.index.name is not None
                or data_frame.index.names[0] is not None):
            save_index = True
        else:
            save_index = False
        data_frame.to_csv(
            path_or_buf=save_path,
            index=save_index,
            header=True,
        )
        return save_path

    def save_json(
            self,
            data: dict,
            folder_name: str,
            file_name: str,
            inner_folders: list[str] = None,
            save_if_exists: bool = True,
            path: Path | None = None,
    ) -> str:
        """
Saves a dictionary as a JSON file in the specified folder.

    This method creates a JSON file from the provided data and saves it 
    in the specified folder. If the file already exists and 
    `save_if_exists` is set to False, the method will return the path 
    of the existing file without overwriting it. If the file does not 
    exist, it will create the necessary directories and save the file.

    Args:
        data (dict): The dictionary data to be saved as a JSON file.
        folder_name (str): The name of the folder where the file will be saved.
        file_name (str): The name of the JSON file to be created.
        inner_folders (list[str], optional): A list of inner folder names to 
            create within the specified folder. Defaults to None.
        save_if_exists (bool, optional): If True, the method will overwrite 
            the existing file. If False, it will not overwrite and will return 
            the existing file path. Defaults to True.
        path (Path | None, optional): An optional path to save the file. 
            If provided, this path will be used instead of constructing one 
            from the folder and file names. Defaults to None.

    Returns:
        str: The path to the saved JSON file.
    """
        save_path = self.get_path(
            folder_name=folder_name,
            file_name=file_name,
            inner_folders=inner_folders,
        ) if path is None else path

        if os.path.isfile(save_path) and not save_if_exists:
            return save_path

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f)

        return save_path

    def get_path(
            self,
            folder_name: str,
            file_name: str,
            inner_folders: list[str] = None,
            to_save: bool = True,
    ) -> Path:
        """
Constructs a file path based on the provided folder and file names.

    This method generates a complete file path by combining the base save or load path,
    the source name, the specified folder name, and any additional inner folders. 
    The final path will include the specified file name.

    Args:
        folder_name (str): The name of the folder to include in the path.
        file_name (str): The name of the file to include in the path.
        inner_folders (list[str], optional): A list of additional inner folder names to include in the path. Defaults to None.
        to_save (bool, optional): A flag indicating whether to use the save path (True) or load path (False). Defaults to True.

    Returns:
        Path: The constructed file path as a Path object.
    """
        path_list = [
            self.save_path if to_save else self.load_path,
            self.source.name,
            folder_name,
        ]
        if inner_folders is not None:
            path_list += inner_folders

        return Path(pjoin(*path_list, file_name))

    def load_features(self, suffix: str | None = None) -> pd.DataFrame:
        features_dataset = self.load(
            folder_name=self.get_name(name=self.data_folder["features"]),
            file_name=self.get_file_name(
                prefix=self.config.features_prefix,
                suffix=suffix
            ),
            get_index=self.has_index["features"],
        )
        return features_dataset

    def save_features(
            self,
            features_handled: pd.DataFrame,
            suffix: str | None = None,
    ) -> str:
        """
Saves the processed features to a specified location.

    This method saves the provided DataFrame of handled features to a designated folder
    with a generated file name based on the class configuration and an optional suffix.

    Args:
        features_handled (pd.DataFrame): The DataFrame containing the processed features to be saved.
        suffix (str | None, optional): An optional suffix to append to the file name. Defaults to None.

    Returns:
        str: The path to the saved file.
    """
        return self.save(
            data_frame=features_handled,
            folder_name=self.get_name(name=self.class_folder),
            file_name=self.get_file_name(
                prefix=self.config.features_prefix,
                suffix=suffix
            ),
        )

    def load_metrics(self, suffix: str | None = None) -> pd.DataFrame:
        metrics_dataset = self.load(
            folder_name=self.get_name(name=self.data_folder["metrics"]),
            file_name=self.get_file_name(
                prefix=self.config.metrics_prefix,
                suffix=suffix
            ),
            get_index=self.has_index["metrics"],
        )
        return metrics_dataset

    def save_metrics(
            self,
            metrics_handled: pd.DataFrame,
            suffix: str | None = None
    ) -> str:
        """
Saves the provided metrics to a specified location.

    This method saves the given metrics DataFrame to a designated folder 
    with a file name that includes a specified prefix and an optional suffix.

    Args:
        metrics_handled (pd.DataFrame): The DataFrame containing the metrics 
            to be saved.
        suffix (str | None, optional): An optional suffix to append to the 
            file name. Defaults to None.

    Returns:
        str: The path to the saved file.
    """
        return self.save(
            data_frame=metrics_handled,
            folder_name=self.get_name(name=self.class_folder),
            file_name=self.get_file_name(
                prefix=self.config.metrics_prefix,
                suffix=suffix
            ),
        )

    def save_samples(
            self,
            data: dict,
            file_name: str,
            inner_folders: list[str] | None = None,
    ) -> str:
        """
Saves the provided data as a JSON file.

    This method saves the given data dictionary to a JSON file with the specified
    file name in a designated folder. It allows for optional inner folder structure
    to be specified.

    Args:
        data (dict): The data to be saved in JSON format.
        file_name (str): The name of the file (without extension) to save the data as.
        inner_folders (list[str] | None, optional): A list of inner folder names to create
            within the main folder. If None, no inner folders will be created.

    Returns:
        str: The path to the saved JSON file.
    """
        return self.save_json(
            data=data,
            folder_name=self.config.sampler_folder,
            file_name=f"{file_name}.json",
            inner_folders=inner_folders,
        )

    def load_samples(
            self,
            file_name: str,
            inner_folders: list[str] | None = None,
    ) -> dict:
        """
Load sample data from a JSON file.

    This method retrieves sample data by loading a JSON file located in a specified
    folder. The file name is constructed using the provided `file_name` parameter,
    and it can optionally navigate through inner folders if specified.

    Args:
        file_name (str): The name of the JSON file to load, without the .json extension.
        inner_folders (list[str] | None): A list of inner folder names to navigate through 
            before accessing the file. If None, the method will not navigate through inner folders.

    Returns:
        dict: The contents of the loaded JSON file as a dictionary.
    """
        return self.load_json(
            folder_name=self.config.sampler_folder,
            file_name=f"{file_name}.json",
            inner_folders=inner_folders,
        )

    @staticmethod
    def get_file_name(prefix: str, suffix: str | None = None):
        """
Generate a file name based on the provided prefix and optional suffix.

    This method constructs a file name by combining the given prefix and suffix.
    If a suffix is provided, it will be included in the file name, separated by 
    double underscores. The file name will have a '.csv' extension.

    Args:
        prefix (str): The prefix to use for the file name.
        suffix (str | None, optional): An optional suffix to append to the file name. 
                                       If not provided, only the prefix will be used.

    Returns:
        str: The generated file name with a '.csv' extension.
    """
        if suffix is not None:
            res = f"{prefix}__{suffix}.csv"
        else:
            res = f"{prefix}.csv"
        return res


class FeaturesHandler(MetadataHandler):
    """
    A class to handle the loading, processing, and saving of feature datasets.

    This class provides methods to load a features dataset, process it, and optionally save 
    the processed dataset. It is designed to streamline the workflow of feature handling in 
    data processing tasks.

    Methods:
        handle_features(load_suffix: str | None, save_suffix: str | None, to_save: bool) -> pd.DataFrame:
            Handles the loading, processing, and optionally saving of feature datasets.

        __handle_features__(features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
            Processes the given features dataset and returns a modified dataset along with handler information.
    """
    def handle_features(
            self,
            load_suffix: str | None = None,
            save_suffix: str | None = None,
            to_save: bool = True
    ) -> pd.DataFrame:
        features_dataset = self.load_features(suffix=load_suffix)
        features_handled, handler_info = self.__handle_features__(
            features_dataset=features_dataset
        )
        if to_save:
            self.save_features(
                features_handled=features_handled,
                suffix=handler_info.info["suffix"] if save_suffix is None else save_suffix,
            )
        return features_handled

    @abstractmethod
    def __handle_features__(self, features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        pass

class MetricsHandler(MetadataHandler):
    """
    A class to handle the loading, processing, and optional saving of metrics data.

    This class provides methods to load a metrics dataset, process it, and save the processed 
    metrics if required. It is designed to facilitate the management of metrics data in a 
    structured manner.

    Methods:
        handle_metrics(load_suffix: str | None, save_suffix: str | None, to_save: bool) -> pd.DataFrame:
            Handles the loading, processing, and optional saving of metrics data.

        __handle_metrics(metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
            Processes the provided metrics dataset and returns a structured output.
    """
    def handle_metrics(
            self,
            load_suffix: str | None = None,
            save_suffix: str | None = None,
            to_save: bool = True
    ) -> pd.DataFrame:
        metrics_dataset = self.load_metrics(suffix=load_suffix)
        metrics_handled, handler_info = self.__handle_metrics__(
            metrics_dataset=metrics_dataset
        )
        if to_save:
            self.save_metrics(
                metrics_handled=metrics_handled,
                suffix=handler_info.info["suffix"] if save_suffix is None else save_suffix,
            )
        return metrics_handled

    @abstractmethod
    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        pass
