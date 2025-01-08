from pathlib import Path
from os.path import join as pjoin

import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

def get_project_path() -> str:
    return str(Path(__file__).parent.parent.parent)

def get_config_path() -> str:
    return pjoin(get_project_path(), "config")


COLORS = {"openml": "green", "uci": "blue", "kaggle": "red"}


def parse_sources(sources, res1, res2):
    openml = [[],[]]
    uci = [[],[]]
    kaggle = [[],[]]

    for i, source in enumerate(sources):
        if source == "openml":
            openml[0].append(res1[i])
            openml[1].append(res2[i])
        elif source == "uci":
            uci[0].append(res1[i])
            uci[1].append(res2[i])
        else:
            kaggle[0].append(res1[i])
            kaggle[1].append(res2[i])
    return openml, uci, kaggle


def mse(source_name, res):
    res = np.array(res)
    res_true = res.copy()
    res_true[1] = res_true[0]

    print(res)
    print(res_true)

    return f"MSE {source_name}: {mean_squared_error(res, res_true)}"


def parse_sb_config() -> dict:
    config_path = pjoin(get_project_path(), "config", "config.yaml")
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def make_path(path_from_config) -> str:
    return pjoin(get_project_path(), *path_from_config.split('/'))


def plot_models(
        model1,
        model2,
        res1,
        res2,
        metric_name,
        task_type,
        ids=None,
        sources=None,
        errs1=None,
        errs2=None
):
    plt.plot()
    plt.plot([0.0, 1.0], [0.0, 1.0], color='grey')
    x = np.array(res1)
    y = np.array(res2)
    if sources is not None:
        colors_sources = [COLORS[i] for i in sources]
        plt.errorbar(x=x, y=y, yerr=errs2, xerr=errs1, color=colors_sources, linestyle='None', marker='^')

        openml, uci, kaggle = parse_sources(sources, res1, res2)

        openml_mse = mse("openml", openml)
        uci_mse = mse("uci", uci)
        kaggle_mse = mse("kaggle", kaggle)

        plt.text(0, 0, f"{openml_mse}\n{uci_mse}\n{kaggle_mse}")
    else:
        plt.errorbar(x=x, y=y, yerr=errs2, xerr=errs1, linestyle='None', marker='^')
    plt.xlabel(f"{model1} {metric_name} score")
    plt.ylabel(f"{model2} {metric_name} score")
    plt.title(task_type)

    if ids is not None:
        for i, txt in enumerate(ids):
            plt.annotate(txt, (x[i], y[i]), fontsize=5)

    plt.show()
    plt.close()
