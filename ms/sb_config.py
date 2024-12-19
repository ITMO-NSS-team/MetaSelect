from os.path import join as pjoin

from ms.utils.sb_utils import make_path


class SBConfig:
    def __init__(
            self,
            use_gpu: bool,
            experiment_config_cpu_path: str,
            experiment_config_gpu_path: str,
            results_path: str,
            tabzilla_path: str
    ):
        self.use_gpu = use_gpu
        self.experiment_config_cpu_path = make_path(experiment_config_cpu_path)
        self.experiment_config_gpu_path = make_path(experiment_config_gpu_path)
        self.results_path = results_path
        self.tabzilla_path = tabzilla_path
        self.datasets_path = pjoin(tabzilla_path, "datasets")