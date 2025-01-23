from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MetadataSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

@dataclass
class TabzillaSource(MetadataSource):
    name: str = "tabzilla"

class SourceBased(ABC):
    @property
    @abstractmethod
    def source(self) -> MetadataSource:
        pass

    @property
    @abstractmethod
    def class_name(self) -> str:
        pass

    @property
    @abstractmethod
    def class_folder(self) -> str:
        pass

    @property
    def name(self) -> str:
        return f"{self.source.name}_{self.class_name}"
