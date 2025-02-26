from typing import Callable
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_validate
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


class MetaModel:
    def __init__(
            self,
            name: str,
            display_name: str,
            model: BaseEstimator,
            params: dict | None = None
    ):
        self.name = name
        self.display_name = display_name
        self.model = model
        self.params = params


    def run(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            splits: dict,
            slices: dict,
            opt_scoring: Callable,
            model_scoring: dict[str, Callable],
            opt_method: str | None,
            opt_cv: int,
            n_trials: int
    ) -> dict[str, dict]:
        print(f"Meta-model: {self.name}")
        default_params = self.model.get_params()
        model_scores = {}
        for model_name in y.columns:
            model_scores[model_name] = {}

        for target_model in y.columns:
            print(f"Training on target model {target_model}")
            self.model.set_params(**default_params)
            if opt_method == "optuna" and self.params:
                print(f"Optimizing hyperparameters for {target_model} using Optuna")
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial: self.objective(
                        trial=trial,
                        x=x,
                        y=y.loc[:, target_model],
                        scoring=opt_scoring,
                        cv=opt_cv
                    ),
                    n_trials=n_trials
                )
                best_params = study.best_params
            elif opt_method == "grid_search" and self.params:
                print(f"Using GridSearchCV for {target_model}")
                grid_search = GridSearchCV(
                    self.model,
                    self.params,
                    cv=opt_cv,
                    scoring=opt_scoring
                )
                grid_search.fit(x, y[target_model])
                best_params = grid_search.best_params_
            else:
                best_params = self.params
            self.model.set_params(**best_params)
            cv_results = self.custom_cv(
                estimator=self.model,
                x_df=x,
                y_df=y.loc[:, target_model],
                splits=splits,
                slices=slices,
                scoring=model_scoring,
                target_model=target_model,
            )
            model_scores[target_model]["cv"] = cv_results
            if best_params is not None:
                model_scores[target_model]["params"] = best_params
            else:
                model_scores[target_model]["params"] = self.model.get_params()

        return model_scores

    def objective(
            self,
            trial: optuna.Trial,
            x: pd.DataFrame,
            y: pd.Series,
            scoring: str,
            cv: int
    ):
        param_grid = {}

        for param, values in self.params.items():
            if all(isinstance(v, int) for v in values):
                param_grid[param] = trial.suggest_int(param, min(values), max(values))
            elif all(isinstance(v, float) for v in values):
                param_grid[param] = trial.suggest_float(param, min(values), max(values))
            elif all(isinstance(v, (int, float)) for v in values):
                param_grid[param] = trial.suggest_float(param, float(min(values)), float(max(values)))
            else:
                param_grid[param] = trial.suggest_categorical(param, values)

        self.model.set_params(**param_grid)
        cv_results = cross_validate(
            estimator=self.model,
            X=x,
            y=y,
            scoring=scoring,
            cv=cv,
            error_score="raise",
        )
        return cv_results["test_score"].mean()

    @staticmethod
    def custom_cv(estimator, x_df, y_df, splits, slices, scoring, target_model) -> dict:
        res = {}
        for score_name, score_func in scoring.items():
            res[f"test_{score_name}"] = []
            for i in splits:
                train = splits[i]["train"]
                test = splits[i]["test"]
                estimator.fit(
                    x_df.iloc[train, :].loc[:, slices[i][target_model]],
                    y_df.iloc[train]
                )
                res[f"test_{score_name}"].append(
                    score_func(
                        estimator=estimator,
                        X=x_df.iloc[test, :].loc[:, slices[i][target_model]],
                        y_true=y_df.iloc[test],
                    )
                )
        return res
