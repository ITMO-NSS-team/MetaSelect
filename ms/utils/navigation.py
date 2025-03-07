import json
import os
from abc import ABC, abstractmethod
from os.path import join as pjoin
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure


def get_project_path() -> str:
    return str(Path(__file__).parent.parent.parent)

def get_config_path() -> str:
    return pjoin(get_project_path(), "config")

def get_prefix(s: str) -> str:
    return s.split("__")[0]

def get_suffix(s: str) -> str:
    return s.split("__")[-1]

def has_suffix(s: str) -> bool:
    return "__" in s


class ResultType(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def save(self, path: Path, data: Any) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> Any:
        pass


class JSONType(ResultType):
    name = "json"

    def save(self, path: Path, data: dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        return data

class CSVType(ResultType):
    name = "csv"

    def save(self, path: Path, data: pd.DataFrame) -> None:
        data.to_csv(
            path_or_buf=path,
            header=True,
        )

    def load(self, path: Path) -> pd.DataFrame:
        data = pd.read_csv(path, index_col=False)
        return data

class PNGType(ResultType):
    name = "png"

    def save(self, path: Path, data: Figure) -> None:
        data.savefig(path)

    def load(self, path: Path) -> Figure:
        return Figure()

result_types = {
    "json": JSONType(),
    "csv": CSVType(),
    "png": PNGType(),
}

def load(path: Path, file_type: str) -> Any:
    return result_types[file_type].load(path=path)

def save(data: Any, path: Path, file_type: str,) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    result_types[file_type].save(path=path, data=data)

def get_path(
        folders: list[str],
        file_name: str,
        file_type: str,
) -> Path:
    path = pjoin(
        get_project_path(),
        *folders,
        f"{file_name}.{file_type}",
    )
    return Path(path)

def rewrite_decorator(func):
    def wrapper(
            self,
            save_path: Path,
            to_rewrite: bool,
            *args,
            **kwargs
    ):
        if not to_rewrite and save_path.exists():
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return func(
            self,
            save_path=save_path,
            *args,
            **kwargs
        )
    return wrapper
