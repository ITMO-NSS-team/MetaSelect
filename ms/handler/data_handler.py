from abc import abstractmethod, ABC
from pathlib import Path

import pandas as pd

from ms.config.navigation_config import NavigationConfig
from ms.handler.data_source import SourceBased
from ms.handler.handler_info import HandlerInfo
from ms.utils.debug import Debuggable
from ms.utils.navigation import load, save, get_path, rewrite_decorator


class DataHandler(SourceBased, Debuggable, ABC):
    def __init__(
            self,
            features_folder: str,
            metrics_folder: str | None = None,
            test_mode: bool = False,
    ):
        super().__init__(
            test_mode=test_mode,
        )
        self._config = NavigationConfig()

        if metrics_folder is None:
            _metrics_folder = features_folder
        else:
            _metrics_folder = metrics_folder
        self._data_folder = {
            "features": features_folder,
            "metrics": _metrics_folder
        }

    @property
    @abstractmethod
    def class_name(self) -> str:
        pass

    @property
    @abstractmethod
    def class_folder(self) -> str:
        pass

    @property
    @abstractmethod
    def class_suffix(self) -> str | None:
        pass

    @property
    def config(self) -> NavigationConfig:
        return self._config

    @config.setter
    def config(self, config: NavigationConfig) -> None:
        self._config = config

    @property
    def data_folder(self) -> dict[str, str]:
        return self._data_folder

    @data_folder.setter
    def data_folder(self, data_folder: dict[str, str]) -> None:
        self._data_folder = data_folder

    @property
    @abstractmethod
    def load_root(self) -> str:
        pass

    @property
    @abstractmethod
    def save_root(self) -> str:
        pass

    def get_path(
            self,
            root_type: str,
            inner_folders: list[str],
            prefix: str,
            file_type: str,
            suffix: str | None = None,
    ) -> Path:
        if root_type == "load":
            root_folder = self.load_root
        elif root_type == "save":
            root_folder = self.save_root
        else:
            raise ValueError("Unknown root type")

        folders = [root_folder, self.source.name]
        folders += [
            self.get_name(name=folder) for folder in inner_folders
        ]
        file_name = self.get_file_name(prefix=prefix, suffix=suffix)
        path = get_path(
            folders=folders,
            file_name=file_name,
            file_type=file_type
        )
        return path

    def get_features_path(
            self,
            root_type: str,
            folder: str,
            suffix: str | None = None
    ) -> Path:
        return self.get_path(
            root_type=root_type,
            inner_folders=[folder],
            prefix=self.config.features_prefix,
            file_type="csv",
            suffix=suffix
        )

    def get_metrics_path(
            self,
            root_type: str,
            folder: str,
            suffix: str | None = None
    ) -> Path:
        return self.get_path(
            root_type=root_type,
            inner_folders=[folder],
            prefix=self.config.metrics_prefix,
            file_type="csv",
            suffix=suffix
        )

    def get_samples_path(
            self,
            root_type: str,
            folders: list[str] | None = None,
            sample_type: str = "slices"
    ) -> str:
        inner_folders = [self.config.sampler_folder]
        if folders is not None:
            inner_folders += folders
        return self.get_path(
            root_type=root_type,
            inner_folders=inner_folders,
            prefix=sample_type,
            file_type="json"
        )

    def load_features(
            self,
            suffix: str | None = None,
            path: Path | None = None,
            folder: str | None = None,
    ) -> pd.DataFrame:
        path = self.get_features_path(
            root_type="load",
            folder=self.data_folder["features"] if folder is None else folder,
            suffix=suffix,
        ) if path is None else path
        df = load(
            path=path,
            file_type="csv",
        )
        if df.columns[0] == self.config.dataset_name:
            df.set_index(self.config.dataset_name, drop=True, inplace=True)
        return df

    def save_features(
            self,
            features: pd.DataFrame,
            suffix: str | None = None,
            path: Path | None = None,
            folder: str | None = None,
    ) -> None:
        path = self.get_features_path(
            root_type="save",
            folder=self.data_folder["features"] if folder is None else folder,
            suffix=suffix,
        ) if path is None else path
        save(
            data=features,
            path=path,
            file_type="csv",
        )

    def load_metrics(
            self,
            suffix: str | None = None,
            path: Path | None = None,
    ) -> pd.DataFrame:
        path = self.get_metrics_path(
            root_type="load",
            folder=self.data_folder["metrics"],
            suffix=suffix,
        ) if path is None else path
        df = load(
            path=path,
            file_type="csv",
        )
        if df.columns[0] == self.config.dataset_name:
            df.set_index(self.config.dataset_name, drop=True, inplace=True)
        return df

    def save_metrics(
            self,
            metrics: pd.DataFrame,
            suffix: str | None = None,
            path: Path | None = None,
    ) -> None:
        path = self.get_metrics_path(
            root_type="save",
            folder=self.data_folder["metrics"],
            suffix=suffix,
        ) if path is None else path
        save(
            data=metrics,
            path=path,
            file_type="csv",
        )

    def load_samples(
            self,
            folders: list[str],
            sample_type: str = "slices", # or splits
            path: Path | None = None,
    ) -> dict:
        json_path = self.get_samples_path(
            root_type="load",
            folders=folders,
            sample_type=sample_type,
        ) if path is None else path
        return load(path=json_path, file_type="json")

    def save_samples(
            self,
            samples: dict,
            sample_type: str = "slices", # or splits
            folders: list[str] | None = None,
            path: Path | None = None,
    ) -> dict:
        json_path = self.get_samples_path(
            root_type="save",
            folders=folders,
            sample_type=sample_type,
        ) if path is None else path
        return save(data=samples, path=json_path, file_type="json")

    @staticmethod
    def get_file_name(prefix: str, suffix: str | None = None):
        if suffix is not None:
            res = f"{prefix}__{suffix}"
        else:
            res = f"{prefix}"
        return res


class FeaturesHandler(DataHandler):
    def handle_features(
            self,
            load_suffix: str | None = None,
            save_suffix: str | None = None,
            to_rewrite: bool = False,
    ) -> pd.DataFrame:
        load_path = self.get_features_path(
            root_type="load",
            folder=self.data_folder["features"],
            suffix=load_suffix,
        )
        save_path = self.get_features_path(
            root_type="save",
            folder=self.class_folder,
            suffix=save_suffix if save_suffix is not None
            else self.class_suffix
        )
        return self.wrap_features(
            load_path=load_path,
            save_path=save_path,
            to_rewrite=to_rewrite,
        )

    @rewrite_decorator
    def wrap_features(
            self,
            load_path: Path,
            save_path: Path,
    ) -> pd.DataFrame:
        features_dataset = self.load_features(path=load_path)
        features_handled, handler_info = self.__handle_features__(
            features_dataset=features_dataset
        )
        save(
            data=features_handled,
            path=save_path,
            file_type="csv"
        )
        return features_handled

    @abstractmethod
    def __handle_features__(self, features_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        pass

class MetricsHandler(DataHandler):
    def handle_metrics(
            self,
            load_suffix: str | None = None,
            save_suffix: str | None = None,
            to_rewrite: bool = False,
    ) -> pd.DataFrame:
        load_path = self.get_metrics_path(
            root_type="load",
            folder=self.data_folder["metrics"],
            suffix=load_suffix,
        )
        save_path = self.get_metrics_path(
            root_type="save",
            folder=self.class_folder,
            suffix=save_suffix if save_suffix is not None
            else self.class_suffix
        )
        return self.wrap_metrics(
            load_path=load_path,
            save_path=save_path,
            to_rewrite=to_rewrite,
        )

    @rewrite_decorator
    def wrap_metrics(
            self,
            load_path: Path,
            save_path: Path,
    ) -> pd.DataFrame:
        metrics_dataset = self.load_metrics(path=load_path)
        metrics_handled, handler_info = self.__handle_metrics__(
            metrics_dataset=metrics_dataset
        )
        save(
            data=metrics_handled,
            path=save_path,
            file_type="csv"
        )
        return metrics_handled

    @abstractmethod
    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> tuple[pd.DataFrame, HandlerInfo]:
        pass
