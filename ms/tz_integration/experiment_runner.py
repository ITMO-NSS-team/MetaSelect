import os

from tqdm import tqdm
from ms.tz_integration.datasets_parser import parse_datasets

from ms.utils.sb_utils import parse_sb_config, make_path
from os.path import join as pjoin
from ms.tz_integration.tz_calls import get_experiment_args, run_experiment

def run_all_experiments(models: list[str], datasets: list[str], use_gpu: bool = True):
    sb_config = parse_sb_config()
    experiment_config_path = sb_config['experiment_config_gpu_path'] if use_gpu \
        else sb_config['experiment_config_cpu_path']
    experiment_config_path = make_path(experiment_config_path)

    experiment_args = get_experiment_args(config_path=experiment_config_path)
    results_path = make_path(sb_config["results_path"])
    datasets_path = make_path(sb_config['datasets_path'])

    for model in tqdm(models, desc="model"):
        result_model_path = pjoin(results_path, model)
        os.makedirs(result_model_path, exist_ok=True)
        for dataset in tqdm(datasets, desc="dataset"):
            result_dataset_path = pjoin(result_model_path, dataset)
            dataset_path = pjoin(datasets_path, dataset)
            os.makedirs(result_dataset_path, exist_ok=True)

            print()
            if len(os.listdir(result_dataset_path)) != 0:
                print(f"Already used. Skipping {dataset}")
                continue
            else:
                print(f"Analyzing {dataset} on {model}")

            experiment_args.output_dir = result_dataset_path
            run_experiment(
                experiment_args=experiment_args,
                model_name=model,
                dataset_dir=dataset_path
            )


if __name__ == "__main__":
    models_list = ["rtdl_MLP", "rtdl_ResNet", "rtdl_FTTransformer"]
    datasets_dict = parse_datasets()
    datasets_list = list(datasets_dict.keys())
    run_all_experiments(models=models_list, datasets=datasets_list, use_gpu=False)
