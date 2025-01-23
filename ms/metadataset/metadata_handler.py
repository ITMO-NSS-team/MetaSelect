import os
from abc import abstractmethod, ABC

import pandas as pd

from ms.metadataset.metadata_config import MetadataConfig
from ms.metadataset.metadata_source import SourceBased
from ms.utils.debug_utils import Debuggable
from ms.utils.ms_utils import pjoin


class MetadataHandler(SourceBased, Debuggable, ABC):
    def __init__(
            self,
            features_folder: str,
            metrics_folder: str | None = None,
    ):
        super().__init__()
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

    def load(
            self,
            folder_name: str,
            file_name: str,
            get_index: bool = True
    ) -> pd.DataFrame:
        data_path = pjoin(
            self.config.data_path,
            self.source.name,
            folder_name,
            file_name,
        )
        if get_index:
            data_frame = pd.read_csv(data_path, index_col=0)
        else:
            data_frame = pd.read_csv(data_path)

        return data_frame

    def save(
            self,
            data_frame: pd.DataFrame,
            folder_name: str,
            file_name: str
    ) -> str:
        save_path = pjoin(
            self.config.data_path,
            self.source.name,
            folder_name,
        )
        os.makedirs(save_path, exist_ok=True)
        save_path = pjoin(save_path, file_name)
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


class FeaturesHandler(MetadataHandler):

    def handle_features(self) -> str:
        features_dataset = self.load(
            folder_name=self.get_name(name=self.data_folder["features"]),
            file_name=self.config.features_name,
            get_index=self.has_index["features"],
        )
        features_handled = self.__handle_features__(
            features_dataset=features_dataset
        )
        return self.save(
            data_frame=features_handled,
            folder_name=self.get_name(name=self.class_folder),
            file_name=self.config.features_name
        )

    @abstractmethod
    def __handle_features__(self, features_dataset: pd.DataFrame) -> pd.DataFrame:
        pass

class MetricsHandler(MetadataHandler):

    def handle_metrics(self) -> str:
        metrics_dataset = self.load(
            folder_name=self.get_name(name=self.data_folder["metrics"]),
            file_name=self.config.metrics_name,
            get_index=self.has_index["metrics"],
        )
        metrics_handled = self.__handle_metrics__(
            metrics_dataset=metrics_dataset
        )
        return self.save(
            data_frame=metrics_handled,
            folder_name=self.get_name(name=self.class_folder),
            file_name=self.config.metrics_name
        )

    @abstractmethod
    def __handle_metrics__(self, metrics_dataset: pd.DataFrame) -> pd.DataFrame:
        pass
