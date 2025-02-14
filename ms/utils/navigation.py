from os.path import join as pjoin
from pathlib import Path


def get_project_path() -> str:
    return str(Path(__file__).parent.parent.parent)

def get_config_path() -> str:
    return pjoin(get_project_path(), "config")
