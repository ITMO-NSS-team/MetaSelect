from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MetadataSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """
Generate a formatted name based on the source name and class name.

    This method constructs a string that combines the name of the source
    object and the class name, separated by an underscore.

    Returns:
        str: A formatted string in the form of 'source_name_class_name'.
    """
        pass

@dataclass
class TabzillaSource(MetadataSource):
    name: str = "tabzilla"

class SourceBased(ABC):
    """
    A class that represents a source-based entity, providing methods to retrieve 
    metadata and class-related information.

    Methods:
        source: Retrieve the metadata source.
        class_name: Retrieve the name of the class.
        class_folder: Retrieve the class folder name.
        name: Generate a formatted name based on the source name and class name.

    Attributes:
        None

    The SourceBased class provides functionality to access metadata sources and 
    class information, allowing for the generation of formatted names that 
    incorporate both the source and class details.
    """
    @property
    @abstractmethod
    def source(self) -> MetadataSource:
        """
Retrieve the metadata source.

    This method returns the current metadata source associated with the instance.

    Returns:
        MetadataSource: The metadata source object.
    """
        pass

    @property
    @abstractmethod
    def class_name(self) -> str:
        """
Retrieve the name of the class.

    This method returns the name of the class to which the instance belongs.

    Returns:
        str: The name of the class as a string.
    """
        pass

    @property
    @abstractmethod
    def class_folder(self) -> str:
        """
Retrieve the class folder name.

    This method is intended to return the name of the folder associated with the class instance.

    Returns:
        str: The name of the class folder.
    """
        pass

    @property
    def name(self) -> str:
        return f"{self.source.name}_{self.class_name}"
