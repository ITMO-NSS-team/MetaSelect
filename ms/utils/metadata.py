import numpy as np
import pandas as pd


def remove_constant_features(features_dataset: pd.DataFrame) -> None:
    features_to_remove = []
    for feature in features_dataset.columns:
        x_i = features_dataset[feature].to_numpy()
        if np.all(x_i == x_i[0]):
            features_to_remove.append(feature)
    features_dataset.drop(features_to_remove, axis="columns", inplace=True)
