import pandas as pd


class MetaDataset:
    def __init__(
            self,
            features_path: str,
            eval_path: str,
            index_name: str = "dataset_name",
            alg_name: str = "alg_name",
            save_path: str | None = None,
    ):
        self.features_dataset = pd.read_csv(features_path, index_col=0)
        self.eval_dataset = pd.read_csv(eval_path, index_col=0)
        self.index_name = index_name
        self.alg_name = alg_name
        self.save_path = save_path
