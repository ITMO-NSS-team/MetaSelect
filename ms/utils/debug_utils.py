from abc import ABC


class Debuggable(ABC):
    def __init__(self, test_mode: bool = False):
        self._test_mode = test_mode

    @property
    def test_mode(self) -> bool:
        return self._test_mode

    @test_mode.setter
    def test_mode(self, value: bool) -> None:
        self._test_mode = value

    def get_name(self, name: str) -> str:
        return f"{name}_test" if self.test_mode else name