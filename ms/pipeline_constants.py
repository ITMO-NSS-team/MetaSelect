from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from ms.handler.metadata_source import TabzillaSource
from ms.metaresearch.meta_model import MetaModel
from ms.metaresearch.selectors.base import *
from ms.metaresearch.selectors.causal import *
from ms.metaresearch.selectors.model_based import *
from ms.metaresearch.selectors.model_free import *

seed = 42

md_source = TabzillaSource()

k_fold_splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
train_test_slitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)

corr = CorrelationSelector(md_source=md_source)
f_val = FValueSelector(md_source=md_source)
mi = MutualInfoSelector(md_source=md_source)
xgb = XGBSelector(md_source=md_source)
lasso = LassoSelector(md_source=md_source)
te = TESelector(md_source=md_source)
base = BaseSelector(md_source=md_source)

mean_cols = ["test_b_acc_mean", "test_f1_mean", "test_roc_mean"]
std_cols = ["test_b_acc_std", "test_f1_std", "test_roc_std"]
target_models = ["LR", "RF", "XGB", "MLP", "FTT", "RN"]

all_handlers = {
        "corr": (CorrelationSelector, corr),
        "f_val": (FValueSelector, f_val),
        "mi": (MutualInfoSelector, mi),
        "xgb": (XGBSelector, xgb),
        "lasso": (LassoSelector, lasso),
        "te": (TESelector, te),
        "base": (BaseSelector, base),
}

# selectors_to_use = ["base", "corr", "f_val", "mi", "xgb", "lasso", "te"]
features_suffixes = ["power"]
# metrics_suffixes = ["perf_abs", "perf_rel", "diff"]

grid_scoring = "b_acc"
model_scoring = {
        'b_acc': make_scorer(balanced_accuracy_score),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc': make_scorer(
            roc_auc_score,
            average='weighted',
            max_fpr=None,
            multi_class="ovo",
            response_method="predict_proba"
        ),
}

lr_mm = MetaModel(
        name="logreg",
        display_name="Logistic Regression",
        model=LogisticRegression(),
        params={
            "penalty": "l2",
            "C": 0.05,
            "solver": "lbfgs",
        }
)

mlp_mm = MetaModel(
        name="mlp",
        display_name="MLP",
        model=MLPClassifier(),
        params={
            "hidden_layer_sizes": 25,
            "activation": "logistic",
            "solver": "lbfgs",
            "alpha": 0.05,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.05,
            "max_iter": 100,
        },
)

xgb_mm = MetaModel(
        name="xgb",
        display_name="XGBoost",
        model=XGBClassifier(),
        params={
            'max_depth': 7,
            'learning_rate': 0.1,
            'n_estimators': 50,
            "eval_metric": "merror",
        },
)

knn_mm = MetaModel(
        name="knn",
        display_name="KNN",
        model=KNeighborsClassifier(),
        params={
            "n_neighbors": 6,
            "weights": "uniform",
            "leaf_size": 40,
            "algorithm": "auto",
            "p": 1,
        }
)
