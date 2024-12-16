from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pymfe.mfe import MFE


class MfHandler:
    def __init__(
            self,
            groups: list[str] | None = None,
            summary_functions: list[str] | None = None,
            score: str | None = None,
            random_state: int | None = None,
            transform_num: bool = False,
            transform_cat: str | None = None,
            rescale: str | None = None,
            cat_cols: list[str] | str = "auto",
            suppress_warnings: bool = False,
    ):
        self.groups = groups
        self.summary_functions = summary_functions
        self.score = score
        self.random_state = random_state
        self.transform_num = transform_num
        self.transform_cat = transform_cat
        self.cat_cols = cat_cols
        self.rescale = rescale
        self.suppress_warnings = suppress_warnings

    def extract_mf(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        mfe = MFE(
            groups=self.groups,
            summary=self.summary_functions,
            score=self.score,
            random_state=self.random_state,
            suppress_warnings=self.suppress_warnings,
        )
        mfe.fit(
            X=x,
            y=y,
            transform_num=self.transform_num,
            transform_cat=self.transform_cat,
            rescale=self.rescale,
            cat_cols=self.cat_cols,
            suppress_warnings=self.suppress_warnings,
        )
        mf = mfe.extract(suppress_warnings=self.suppress_warnings)
        return mf


class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


@dataclass
class Feature:
    feature_name: str
    type: FeatureType
    description: str | None = None
