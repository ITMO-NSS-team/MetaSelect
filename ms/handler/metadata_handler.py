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
    def __init__(
            self,
            features_folder: str,
            metrics_folder: str | None = None,
            test_mode: bool = False,
    ):
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
        return self._config

    @config.setter
    def config(self, config: MetadataConfig) -> None:
        self._config = config

    @property
    def data_folder(self) -> dict[str, str]:
        return self._data_folder

    @data_folder.setter
    def data_folder(self, data_folder: dict[str, str]) -> None:
        self._data_folder = data_folder

    @property
    @abstractmethod
    def has_index(self) -> dict[str, bool]:
        pass

    @property
    def load_path(self) -> str:
        return self.config.data_path

    @property
    @abstractmethod
    def save_path(self) -> str:
        pass

    def load(
            self,
            folder_name: str,
            file_name: str,
            get_index: bool = True,
            path: str | None = None,
    ) -> pd.DataFrame:
        data_path = pjoin(
            self.load_path,
            self.source.name,
            folder_name,
            file_name,
        ) if path is None else path
        if get_index:
            data_frame = pd.read_csv(data_path, index_col=0)
        else:
            data_frame = pd.read_csv(data_path)

        return data_frame

    def save(
            self,
            data_frame: pd.DataFrame,
            folder_name: str,
            file_name: str,
            inner_folders: list[str] = None,
            save_if_exists: bool = True,
    ) -> str:
        save_path = self.get_save_path(
            folder_name=folder_name,
            file_name=file_name,
            inner_folders=inner_folders,
        )

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

    def get_save_path(
            self,
            folder_name: str,
            file_name: str,
            inner_folders: list[str] = None,
    ) -> Path:
        path_list = [
            self.save_path,
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
        return self.save(
            data_frame=metrics_handled,
            folder_name=self.get_name(name=self.class_folder),
            file_name=self.get_file_name(
                prefix=self.config.metrics_prefix,
                suffix=suffix
            ),
        )

    @staticmethod
    def get_file_name(prefix: str, suffix: str | None = None):
        if suffix is not None:
            res = f"{prefix}__{suffix}.csv"
        else:
            res = f"{prefix}.csv"
        return res


class FeaturesHandler(MetadataHandler):
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
