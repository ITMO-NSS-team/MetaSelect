from dataclasses import dataclass


@dataclass
class HandlerInfo:
    def __init__(self, suffix: str | None = None):
        self.info = {"suffix": suffix}
