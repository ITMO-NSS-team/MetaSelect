from dataclasses import dataclass
from typing import Any


@dataclass
class HandlerInfo:
    def __init__(
            self,
            suffix: str | None = None,
            feature_index: int | None = None,
            metrics_index: int | None = None,
    ):
        self.info = {
            "suffix": suffix,
            "feature_index": feature_index,
            "metrics_index": metrics_index,
        }

    def add_info(self, key: str, value: Any) -> None:
        if self.info.get(key) is None:
            self.info[key] = value

    def get_info(self, key: str) -> Any:
        return self.info[key]
