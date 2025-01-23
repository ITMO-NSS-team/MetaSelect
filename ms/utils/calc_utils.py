class AbsRangeStorage:
    def __init__(self, default_init: bool = True):
        self.storage: dict[int, tuple[float, float]] = {}
        if default_init:
            self.add_range(key=0, right_value=0.0, left_value=0.5)
            self.add_range(key=1, right_value=0.5, left_value=1.0)

    def add_range(self, key: int, right_value: float, left_value: float) -> None:
        self.storage[key] = (right_value, left_value)