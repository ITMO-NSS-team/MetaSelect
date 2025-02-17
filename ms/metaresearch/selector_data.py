import pandas as pd


class SelectorData:
    def __init__(
            self,
            name: str,
            features_suffix: str,
            metrics_suffix: str,
            features: dict[int, list[str]] | None = None,
    ):
        self.name = name
        self.features_suffix = features_suffix
        self.metrics_suffix = metrics_suffix
        self.features = features

    def get_features(self, x: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        if self.features is None:
            return x
        return x.loc[:, self.features[sample_size]]
