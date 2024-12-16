import numpy as np


class TargetBuilder:
    def __init__(
            self,
            model_1: str,
            model_2: str,
            metric: str,
            result_type: str = "delta", # or "label"
            to_abs: bool = False,
            atol: float = 1e-3,
    ):
        self.model_1 = model_1
        self.model_2 = model_2
        self.metric = metric
        self.result_type = result_type
        self.to_abs = to_abs
        self.atol = atol
        self.name = f"target_{result_type}_{metric}_{model_1}_{model_2}"

    def get_result(self, a: float, b: float) -> float | int:
        return self.get_delta(a=a, b=b) if self.result_type == "delta" \
            else self.get_label(a=a, b=b)

    def get_delta(self, a: float, b: float) -> float:
        return np.abs(a - b) if self.to_abs else a - b

    def get_label(self, a: float, b: float) -> int:
        label = 0
        if np.isclose(a, b, atol=self.atol):
            pass
        elif a > b:
            label = 1
        else:
            label = 2
        return label
