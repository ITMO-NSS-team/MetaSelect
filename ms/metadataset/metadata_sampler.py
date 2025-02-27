from random import sample

import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit

from ms.handler.metadata_handler import MetadataHandler
from ms.handler.metadata_source import MetadataSource, TabzillaSource


class MetadataSampler(MetadataHandler):
    """
    A class for sampling metadata and generating training/testing splits for datasets.

    This class provides functionality to manage and sample metadata from a specified 
    source, allowing for the generation of training and testing datasets based on 
    defined features and metrics. It includes methods for retrieving class information, 
    checking the availability of features, and saving sample data.

    Methods:
        - class_name: Returns the name of the class.
        - class_folder: Retrieves the folder path for the sampler configuration.
        - source: Retrieves the metadata source.
        - has_index: Checks the availability of features and metrics.
        - save_path: Retrieves the path to the resources.
        - __init__: Initializes an instance of the class.
        - sample_data: Generates and saves sample data splits for specified features.
        - make_samples: Generates and saves sample data based on the provided feature dataset.
        - sample_uninformative: Generates and saves samples from a DataFrame based on specified percentages.

    """
    @property
    def class_name(self) -> str:
        """
Return the name of the class.

    This method returns a string that represents the name of the class 
    associated with the instance.

    Returns:
        str: The name of the class, which is "features_sampler".
    """
        return "features_sampler"

    @property
    def class_folder(self) -> str:
        """
Retrieve the folder path for the sampler configuration.

    This method accesses the configuration object associated with the
    instance and returns the path to the folder where sampler files
    are stored.

    Returns:
        str: The path to the sampler folder as defined in the configuration.
    """
        return self.config.sampler_folder

    @property
    def source(self) -> MetadataSource:
        """
Retrieve the metadata source.

    This method returns the metadata source associated with the current instance.

    Returns:
        MetadataSource: The metadata source object.
    """
        return self._md_source

    @property
    def has_index(self) -> dict:
        """
Check the availability of features and metrics.

    This method returns a dictionary indicating whether features and metrics are available.

    Returns:
        dict: A dictionary with keys 'features' and 'metrics', both set to True.
    """
        return {
            "features": True,
            "metrics": True,
        }

    @property
    def save_path(self) -> str:
        """
Retrieve the path to the resources.

    This method returns the path to the resources defined in the configuration.

    Returns:
        str: The path to the resources.
    """
        return self.config.resources

    def __init__(
            self,
            md_source: MetadataSource,
            splitter: KFold | ShuffleSplit,
            start: int = 20,
            step: int = 20,
            n_iter: int = 5,
            features_folder: str = "preprocessed",
            metrics_folder: str | None = "preprocessed",
            test_mode: bool = False,
    ):
        """
Initializes an instance of the class.

    This constructor sets up the necessary parameters for the class, including
    the metadata source, data splitting strategy, and configuration for feature
    and metrics storage.

    Args:
        md_source (MetadataSource): The source of metadata used for processing.
        splitter (KFold | ShuffleSplit): The strategy for splitting the dataset.
        start (int, optional): The starting point for processing. Defaults to 20.
        step (int, optional): The step size for processing. Defaults to 20.
        n_iter (int, optional): The number of iterations for processing. Defaults to 5.
        features_folder (str, optional): The folder where features are stored. Defaults to "preprocessed".
        metrics_folder (str | None, optional): The folder where metrics are stored. Defaults to "preprocessed".
        test_mode (bool, optional): Flag indicating whether to run in test mode. Defaults to False.

    Returns:
        None: This method does not return a value.
    """
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source
        self.splitter = splitter
        self.start = start
        self.step = step
        self.n_iter = n_iter

    def sample_data(
            self,
            feature_suffixes: list[str],
            target_suffix: str,
            rewrite: bool = False,
            are_additional: bool = False,
            percents: list[float] | None = None,
    ) -> None:
        """
Generates and saves sample data splits for specified features.

    This method processes a list of feature suffixes to create training and testing splits 
    for the corresponding datasets. It checks if the splits already exist and can optionally 
    rewrite them. Additionally, it can create uninformative samples based on the provided 
    parameters.

    Args:
        feature_suffixes (list[str]): A list of suffixes for the feature datasets to be processed.
        target_suffix (str): The suffix for the target dataset used for metrics.
        rewrite (bool, optional): If True, existing splits will be overwritten. Defaults to False.
        are_additional (bool, optional): If True, additional uninformative samples will be created. 
                                         Defaults to False.
        percents (list[float] | None, optional): A list of percentages to use when creating 
                                                  uninformative samples. Defaults to None.

    Returns:
        None: This method does not return a value. It saves the generated splits to disk.
    """
        for feature_suffix in feature_suffixes:
            splits_dict = {}
            splits_path = self.get_path(
                folder_name=self.config.sampler_folder,
                file_name=f"{self.config.splits_prefix}.json",
                inner_folders=[feature_suffix],
            )
            if not rewrite and splits_path.exists():
                continue

            x_df = self.load_features(suffix=feature_suffix)
            y_df = self.load_metrics(suffix=target_suffix)

            data_split = self.splitter.split(x_df, y_df)

            for i, (train, test) in enumerate(data_split):
                splits_dict[i] = {
                    "train": list(map(int, train)),
                    "test": list(map(int, test)),
                }

            self.save_samples(
                data=splits_dict,
                file_name=f"{self.config.splits_prefix}",
                inner_folders=[feature_suffix],
            )

            if are_additional:
                self.sample_uninformative(
                    add_suffix=feature_suffix,
                    percents=percents,
                    rewrite=rewrite,
                    x_df=x_df
                )
            else:
                self.make_samples(
                    processed_suffix=feature_suffix,
                    feature_dataset=x_df,
                    rewrite=rewrite,
                )


    def make_samples(
            self,
            processed_suffix: str,
            feature_dataset: pd.DataFrame,
            rewrite: bool = False,
    ) -> None:
        """
Generates and saves sample data based on the provided feature dataset.

    This method creates a dictionary of samples from the given feature dataset, 
    where the number of samples is determined by a specified range. The samples 
    are saved to a JSON file in a designated folder. If the file already exists 
    and the `rewrite` parameter is set to False, the method will not overwrite 
    the existing file.

    Args:
        processed_suffix (str): A suffix used to create the path for saving 
            the samples.
        feature_dataset (pd.DataFrame): A pandas DataFrame containing the 
            features from which samples will be drawn.
        rewrite (bool, optional): A flag indicating whether to overwrite the 
            existing samples file. Defaults to False.

    Returns:
        None: This method does not return any value. It saves the generated 
        samples to a file.
    """
        save_path = self.get_path(
            folder_name=self.config.sampler_folder,
            file_name=f"{self.config.slices_prefix}.json",
            inner_folders=[processed_suffix],
        )
        if not rewrite and save_path.exists():
            return
        samples_dict = {}

        f_num = feature_dataset.shape[1]
        samples_range = list(range(self.start, f_num, self.step))

        f_cols = [i for i in range(f_num)]
        if samples_range[-1] != f_num:
            samples_range.append(f_num)

        for n_samples in samples_range:
            samples_dict[n_samples] = {}
            for i in range(self.n_iter):
                samples = sample(f_cols, n_samples)
                samples_dict[n_samples][i] = samples

        self.save_samples(
            data=samples_dict,
            file_name=f"{self.config.slices_prefix}",
            inner_folders=[processed_suffix]
        )

    def sample_uninformative(
            self,
            x_df: pd.DataFrame,
            add_suffix: str,
            percents: list[float],
            rewrite: bool = False,
    ) -> None:
        """
Generates and saves samples from a DataFrame based on specified percentages.

    This method samples data from the provided DataFrame (`x_df`) by selecting columns 
    that match a specified suffix (`add_suffix`). It creates samples based on the 
    percentages provided in the `percents` list and saves the results to a JSON file. 
    If the file already exists and `rewrite` is set to False, the method will not 
    overwrite the existing file.

    Args:
        x_df (pd.DataFrame): The DataFrame from which to sample data.
        add_suffix (str): The suffix used to identify which columns to sample from.
        percents (list[float]): A list of percentages indicating the proportion of 
            data to sample for each iteration.
        rewrite (bool, optional): A flag indicating whether to overwrite the existing 
            file if it exists. Defaults to False.

    Returns:
        None: This method does not return any value. It saves the sampled data to a file.
    """
        save_path = self.get_path(
            folder_name=self.config.sampler_folder,
            file_name=f"{self.config.addition_prefix}.json",
            inner_folders=[add_suffix],
        )
        if not rewrite and save_path.exists():
            return
        samples_dict = {}
        additional_indexes = []
        for i, f in enumerate(list(x_df.columns)):
            if f.split("___")[0] == add_suffix:
                additional_indexes.append(i)

        for i, percent in enumerate(percents):
            sample_size = int(len(additional_indexes) * percent)
            samples_dict[i] = {}
            for j in range (self.n_iter):
                samples_dict[i][j] = sample(additional_indexes, sample_size)

        self.save_samples(
            data=samples_dict,
            file_name=f"{self.config.addition_prefix}",
            inner_folders=[add_suffix]
        )


if __name__ == "__main__":
    k_fold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    train_test_slitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    f_sampler = MetadataSampler(
        md_source=TabzillaSource(),
        splitter=k_fold_splitter,
        features_folder="preprocessed",
        metrics_folder="preprocessed",
        test_mode=False
    )
    f_sampler.sample_data(
        feature_suffixes=["power"],
        target_suffix="perf_abs",
        rewrite=True
    )
