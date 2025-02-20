from os.path import join as pjoin
from pathlib import Path


def get_project_path() -> str:
    return str(Path(__file__).parent.parent.parent)

def get_config_path() -> str:
    return pjoin(get_project_path(), "config")

def get_prefix(s: str) -> str:
    return s.split("__")[0]

def get_suffix(s: str) -> str:
    return s.split("__")[-1]

def has_suffix(s: str) -> bool:
    return "__" in s
