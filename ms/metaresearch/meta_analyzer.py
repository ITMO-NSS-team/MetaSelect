from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


class FeatureAnalyzer(ABC):
    def __init__(self, default_target: str = "new") -> None:
        self.default_target = default_target

    def analyze(
            self,
            features: pd.DataFrame,
            targets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        has_raw_target = True if targets.get("raw") is not None else False
        has_new_target = True if targets.get("new") is not None else False
        if not has_raw_target and not has_new_target:
            raise ValueError("No targets provided")

        return self.__analyze__(
            features=features,
            targets=targets,
        )

    @abstractmethod
    def __analyze__(
            self,
            features: pd.DataFrame,
            targets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        pass

class CorrelationAnalyzer(FeatureAnalyzer):
    def __init__(
            self,
            default_target: str = "new",
            corr_func: str = "pearson",
    ) -> None:
        super().__init__(default_target)
        self.corr_func = corr_func

    def __analyze__(
            self,
            features: pd.DataFrame,
            targets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        corr_res = pd.DataFrame(index=features.columns)
        target = targets[self.default_target]

        for i in range(len(target.columns)):
            cur_model = target.iloc[:, i]
            corr = features.corrwith(
                cur_model,
                method=self.corr_func,
            )
            corr_res[f"{target.columns[i]}"] = corr

        return corr_res


class InfoAnalyzer(FeatureAnalyzer):
    def __init__(self, default_target: str = "new") -> None:
        super().__init__(default_target)

    def __analyze__(
            self,
            features: pd.DataFrame,
            targets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        info_res = pd.DataFrame(index=features.columns)
        target = targets[self.default_target]

        for i in range(len(target.columns)):
            cur_model = target.iloc[:, i]
            info = mutual_info_classif(
                features.to_numpy(),
                cur_model.to_numpy(),
            )
            info_res[f"{target.columns[i]}"] = info

        return info_res


class CausalAnalyzer(FeatureAnalyzer):
    def __init__(self, default_target: str = "new") -> None:
        super().__init__(default_target)

    def __analyze__(
            self,
            features: pd.DataFrame,
            targets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        causal_df = pd.DataFrame(index=features.columns)
        target = targets[self.default_target]

        for i in range(len(target.columns)):
            print(target.columns[i])
            influence_df = self.get_causal(
                features=features,
                target_slice=target.iloc[:, i].to_numpy(),
            )
            causal_df[f"{target.columns[i]}"] = influence_df.abs().mean()

        return causal_df

    @staticmethod
    def get_causal(
            features: pd.DataFrame,
            target_slice: np.ndarray,
    ):
        influence_df = features.copy()
        for idx, f in enumerate(features.columns):
            t = features[[f]].values.ravel()
            covariates = features.drop(
                columns=[f], inplace=False
            ).values
            model_y = RandomForestRegressor()
            model_t = RandomForestRegressor()
            dml = CausalForestDML(model_y=model_y, model_t=model_t)
            dml.fit(Y=target_slice, T=t, X=covariates)
            te = dml.effect(X=covariates)
            influence_df[f] = te
        return influence_df


class SIFTEDAnalyzer(FeatureAnalyzer):
    def __init__(
            self,
            default_target: str = "new",
            k_clusters: int = 10,
            num_trees: int = 50
    ) -> None:
        super().__init__(default_target)
        self.k_clusters = k_clusters
        self.num_trees = num_trees

    def __analyze__(
            self,
            features: pd.DataFrame,
            targets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        y = targets["raw"].to_numpy()
        y_bin = targets["new"].to_numpy()
        x = features.to_numpy()

        rho = np.array([np.corrcoef(x[:, i], y, rowvar=False)[0, 1] for i in range(x.shape[1])])
        p_vals = np.array([0.05 if not np.isnan(r) else np.nan for r in rho])

        rho[np.isnan(rho) | (p_vals > 0.05)] = 0
        sorted_indices = np.argsort(-np.abs(rho))

        filtered_indices = np.zeros(x.shape[1], dtype=bool)
        filtered_indices[sorted_indices[0]] = True
        for i in range(1, x.shape[1]):
            if abs(rho[sorted_indices[i]]) >= 0.3:
                filtered_indices[sorted_indices[i]] = True

        filtered_indices = np.where(filtered_indices)[0]
        x_filtered = x[:, filtered_indices]

        kmeans = KMeans(
            n_clusters=self.k_clusters,
            random_state=0,
            n_init='auto'
        )
        labels = kmeans.fit_predict(x_filtered.T)
        clusters = [np.where(labels == i)[0] for i in range(self.k_clusters)]

        combinations = list(product(*clusters))
        comb_rhos = {}
        for idx, comb in enumerate(combinations):
            comb_rho = np.array([0. for _ in range(len(comb))])
            for i in range(len(comb)):
                comb_rho[i] = rho[comb[i]]
            mean_rho = comb_rho.mean()
            if mean_rho > 0.1:
                comb_rhos[idx] = mean_rho
        comb_rhos = dict(sorted(comb_rhos.items(), key=lambda item: item[1], reverse=True))
        comb_rhos_keys = list(comb_rhos.keys())
        combinations_new = []
        for i in range(100):
            combinations_new.append(combinations[comb_rhos_keys[i]])
        combinations = combinations_new

        errors = []
        for j, comb in enumerate(combinations):
            err = self.cost_function(
                x=x_filtered[:, comb],
                y_bin=y_bin,
            )
            errors.append(err)

        optimal_combination = combinations[np.argmin(errors)]

        filtered_indices = filtered_indices[list(optimal_combination)]
        res_df = pd.DataFrame(
            x[:, filtered_indices],
            columns=filtered_indices,
            index=features.index
        )

        return res_df

    def cost_function(
            self,
            x: np.ndarray,
            y_bin: np.ndarray
    ):
        pca = PCA(n_components=2)
        x_reduced = pca.fit_transform(x)
        errors = []

        for i in range(y_bin.shape[1]):
            clf = RandomForestClassifier(
                n_estimators=self.num_trees,
                oob_score=True,
                n_jobs=-1,
                random_state=0
            )
            clf.fit(x_reduced, y_bin[:, i])
            errors.append(1 - clf.oob_score_)

        return np.mean(errors)
