from abc import ABC, abstractmethod


class MetaAnalyzer(ABC):
    @abstractmethod
    def analyze(self):
        pass


class CausalAnalyzer(MetaAnalyzer):
    def analyze(self):
        pass


class CorrelationAnalyzer(MetaAnalyzer):
    def analyze(self):
        pass


class InfoAnalyzer(MetaAnalyzer):
    def analyze(self):
        pass