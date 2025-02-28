from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import TensorDataset, DataLoader

from ms.handler.data_source import DataSource
from ms.handler.selector_handler import SelectorHandler
from ms.utils.typing import NDArrayFloatT


class TESelector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "te"

    @property
    def class_name(self) -> str:
        return "treatment_effect"

    def __init__(
            self,
            md_source: DataSource,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            quantile_value: float = 0.8,
            test_mode: bool = False,
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.quantile_value = quantile_value

    def handle_data(
            self,
            x: NDArrayFloatT,
            y: NDArrayFloatT,
            features_names: list[str],
    ) -> pd.DataFrame:
        inf_df = pd.DataFrame()
        for i, f_name in enumerate(features_names):
            t = x[:, i]
            covariates = np.concatenate([x[:, :i], x[:, i + 1:]], axis=1)
            model_y = RandomForestRegressor()
            model_t = RandomForestRegressor()
            dml = CausalForestDML(model_y=model_y, model_t=model_t)
            dml.fit(Y=y, T=t, X=covariates)
            te = dml.effect(X=covariates)
            inf_df[f_name] = te
        inf_df = pd.DataFrame(inf_df.mean(), index=features_names, columns=["eff_mean"])
        quantile_eff = inf_df["eff_mean"].abs().quantile(self.quantile_value)

        for i, eff in enumerate(inf_df["eff_mean"].to_numpy()):
            if abs(eff) < quantile_eff:
                inf_df.iloc[i, 0] = None

        return inf_df


class ClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        return self.linear(x)

class CFSelector(SelectorHandler):
    @property
    def class_folder(self) -> str:
        return "cf"
    
    @property
    def class_name(self) -> str:
        return "counterfactual"
    
    def __init__(
        self,
        md_source: DataSource,
        features_folder: str = "preprocessed",
        metrics_folder: str | None = "preprocessed",
        test_mode: bool = False,
        cf_steps: int = 500,
        train_epochs: int = 300,
        dc: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            md_source=md_source,
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self.cf_steps = cf_steps
        self.train_epochs = train_epochs
        self.dc = dc
        self.device = device

    def handle_data(
        self,
        x: NDArrayFloatT,
        y: NDArrayFloatT,
        features_names: list[str],
    ) -> pd.DataFrame:
        X = np.array(x)
        y = np.array(y).flatten()
        num_features = X.shape[1]
        fitness_results = {}

        with ThreadPoolExecutor(max_workers=num_features) as executor:
            futures = {}
            for feat_idx in range(num_features):
                mask = np.zeros(num_features, dtype=bool)
                mask[feat_idx] = True
                future = executor.submit(
                    self._evaluate_feature_subset, X, y, mask
                )
                futures[future] = feat_idx

            for future in as_completed(futures):
                feat_idx = futures[future]
                fitness = future.result()
                fitness_results[feat_idx] = fitness
                

        inf_df = pd.DataFrame.from_dict(
            fitness_results, orient="index", columns=["fitness"]
        )
        new_features_names = [features_names[i] for i in fitness_results]
        inf_df.index = new_features_names
        inf_df.loc[inf_df["fitness"].abs() == 0.0, "fitness"] = None
        return inf_df

    def _evaluate_feature_subset(self, X_train, y_train, feature_mask):
        X_train_masked = X_train[:, feature_mask]
        X_train_tensor = torch.FloatTensor(X_train_masked).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        model = ClassificationModel(feature_mask.sum()).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.train_epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        med = torch.median(X_train_tensor, dim=0).values
        mad = torch.median(torch.abs(X_train_tensor - med) + 1e-6, dim=0).values

        model.eval()
        with torch.no_grad():
            logits = model(X_train_tensor)
            original_classes = torch.argmax(logits, dim=-1)

        target_tensor = ((original_classes + 1) % 2).to(self.device)
        cf_batch = self._generate_counterfactual_batch(
            model, X_train_tensor, target_tensor, mad
        )

        with torch.no_grad():
            logits_cf = model(cf_batch)
            cf_classes = torch.argmax(logits_cf, dim=-1)

        cf_accuracy = (cf_classes == target_tensor).float().mean().item()
        fitness = cf_accuracy
        return fitness

    def _generate_counterfactual_batch(self, model, x_original, target_class, mad):
        x_cf = x_original.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([x_cf], lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.cf_steps):
            optimizer.zero_grad()
            logits = model(x_cf)
            pred_loss = criterion(logits, target_class)
            distance_loss = ((x_cf - x_original).abs() / mad).sum(dim=1).mean()
            loss = pred_loss + self.dc * distance_loss
            loss.backward()
            optimizer.step()

        return x_cf.detach()