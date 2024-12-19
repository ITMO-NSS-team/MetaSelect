import argparse
from pathlib import Path

import ms.tz_integration.tz_import as tz_import

tz_import.run()

from submodules.tabzilla.TabZilla.tabzilla_experiment import main as tabzilla_experiment
from submodules.tabzilla.TabZilla.tabzilla_utils import get_experiment_parser
from submodules.tabzilla.TabZilla.tabzilla_datasets import TabularDataset


def get_experiment_args(config_path: str) -> argparse.Namespace:
    experiment_parser = get_experiment_parser()
    experiment_args = experiment_parser.parse_args(
        args="-experiment_config " + config_path
    )
    return experiment_args


def run_experiment(
        experiment_args: argparse.Namespace,
        model_name: str,
        dataset_dir: str
) -> None:
    tabzilla_experiment(
        experiment_args=experiment_args,
        model_name=model_name,
        dataset_dir=dataset_dir
    )


def get_dataset(dataset_path: str):
    return TabularDataset.read(Path(dataset_path))
