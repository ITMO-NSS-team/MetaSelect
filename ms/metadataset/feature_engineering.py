from random import sample

import numpy as np

from ms.handler.metadata_handler import MetadataHandler
from ms.handler.metadata_source import MetadataSource
from ms.utils.typing import NDArrayFloatT


class FeatureCrafter(MetadataHandler):
    """
    A class for crafting features by applying various transformations to datasets.

    This class provides methods to retrieve metadata, check the availability of features and metrics,
    and perform operations such as adding noise, corrupting features, and generating second-order features.

    Methods:
        source: Retrieve the metadata source.
        class_name: Return the name of the class.
        has_index: Check the availability of features and metrics.
        save_path: Retrieve the path to the resources.
        load_path: Load the resource path from the configuration.
        class_folder: Retrieve the path to the preprocessed folder.
        __init__: Initializes an instance of the class.
        perform: Processes a dataset by adding noise, corrupted features, and second-order features.
        add_random_feature: Generates a random feature based on the specified distribution.
        add_corrupted_feature: Add a corrupted version of a feature based on a specified corruption coefficient.
        add_second_order_feature: Calculates the second-order feature by multiplying two input features.

    Attributes:
        md_source (MetadataSource): The source of metadata used by the instance.
        features_folder (str): The folder path for features.
        metrics_folder (str | None): The folder path for metrics.
        test_mode (bool): A flag indicating whether the instance is in test mode.
    """
    distribution = {
        "normal": np.random.normal,
        "uniform": np.random.uniform,
        "poisson": np.random.poisson,
        "gamma": np.random.gamma,
    }

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
    def class_name(self) -> str:
        """
Return the name of the class.

    This method returns a string that represents the name of the class.

    Returns:
        str: The name of the class, which is "crafter".
    """
        return "crafter"

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

    This method returns the path to the resources defined in the 
    configuration of the instance.

    Returns:
        str: The path to the resources.
    """
        return self.config.resources

    @property
    def load_path(self) -> str:
        """
Load the resource path from the configuration.

    This method retrieves the resource path defined in the configuration object.

    Returns:
        str: The resource path as a string.
    """
        return self.config.resources

    @property
    def class_folder(self) -> str:
        """
Retrieve the path to the preprocessed folder.

    This method accesses the configuration object associated with the instance
    and returns the path to the folder where preprocessed data is stored.

    Returns:
        str: The path to the preprocessed folder.
    """
        return self.config.preprocessed_folder

    def __init__(
            self,
            md_source: MetadataSource,
            features_folder: str = "filtered",
            metrics_folder: str | None = "target",
            test_mode: bool = False,
    ):
        """
Initializes an instance of the class.

    This constructor initializes the object with the provided metadata source,
    features folder, metrics folder, and test mode settings. It also calls the
    superclass constructor with the specified features and metrics folder paths,
    as well as the test mode flag.

    Args:
        md_source (MetadataSource): The source of metadata to be used by the instance.
        features_folder (str, optional): The folder path for features. Defaults to "filtered".
        metrics_folder (str | None, optional): The folder path for metrics. Defaults to "target".
        test_mode (bool, optional): A flag indicating whether the instance is in test mode. Defaults to False.

    Returns:
        None
    """
        super().__init__(
            features_folder=features_folder,
            metrics_folder=metrics_folder,
            test_mode=test_mode,
        )
        self._md_source = md_source

    def perform(
            self,
            features_suffix: str,
            random_percent: float | None = None,
            corrupted_percent: float | None = None,
            second_order_percent: float | None = None,
            dist_name: str = "normal",
            corrupt_coeff: float = 0.5,
    ) -> None:
        """
Processes a dataset by adding noise, corrupted features, and second-order features.

    This method loads a preprocessed dataset, applies specified transformations based on the provided
    percentages for random noise, corrupted features, and second-order features, and then saves the 
    modified dataset.

    Args:
        features_suffix (str): The suffix to append to the feature file name for loading the dataset.
        random_percent (float | None): The percentage of features to add random noise to. If None, no noise is added.
        corrupted_percent (float | None): The percentage of features to corrupt. If None, no features are corrupted.
        second_order_percent (float | None): The percentage of second-order features to create. If None, no second-order features are created.
        dist_name (str): The name of the distribution to use for generating random noise. Defaults to "normal".
        corrupt_coeff (float): The coefficient used for corrupting features. Defaults to 0.5.

    Returns:
        None: This method does not return a value. It saves the modified dataset to a file.
    """
        processed_dataset = self.load(
            folder_name="preprocessed",
            file_name=self.get_file_name(self.config.features_prefix, features_suffix),
        )
        cols = list(processed_dataset.columns)
        rows = list(processed_dataset.index)
        features = len(cols)
        datasets = len(rows)
        features_dataset = self.load_features().loc[rows, cols]
        changed_dataset = processed_dataset.copy()
        if random_percent is not None:
            r_num = int(features * random_percent)
            for i in range(r_num):
                changed_dataset[f"noise___{dist_name}_{i}"] = self.add_random_feature(
                    size=datasets,
                    dist_name=dist_name
                )

        if corrupted_percent is not None:
            c_num = int(features * corrupted_percent)
            sampled = sample(cols, c_num)
            for f_name in sampled:
                feature = features_dataset.loc[:, f_name].to_numpy(copy=True)
                changed_dataset[f"corrupted___{f_name}"] \
                    = self.add_corrupted_feature(
                    feature=feature,
                    corrupt_coeff=corrupt_coeff,
                    dist_name=dist_name
                )

        if second_order_percent is not None:
            so_num = int(features * second_order_percent)
            for i in range(so_num):
                f_name1, f_name2 = sample(cols, 2)
                feature1 = features_dataset.loc[:, f_name1].to_numpy(copy=True)
                feature2 = features_dataset.loc[:, f_name2].to_numpy(copy=True)
                changed_dataset[f"so___{f_name1}_{f_name2}"] \
                    = self.add_second_order_feature(feature_first=feature1, feature_second=feature2)

        percents = [random_percent, corrupted_percent, second_order_percent]
        names = ["noise", "corrupted", "so"]
        save_suffix = ""
        for i, percent in enumerate(percents):
            if percent is not None:
                save_suffix += f"{names[i]}"

        self.save_features(changed_dataset, save_suffix)

    def add_random_feature(
            self,
            size: int,
            dist_name: str
    ) -> NDArrayFloatT:
        """
Generates a random feature based on the specified distribution.

    This method retrieves a random feature from the distribution specified by
    `dist_name` and generates an array of the given `size`.

    Args:
        size (int): The number of random samples to generate.
        dist_name (str): The name of the distribution to use for generating
            the random feature.

    Returns:
        NDArrayFloatT: An array of random samples generated from the specified
        distribution.
    """
        return self.distribution[dist_name](size=size)

    def add_corrupted_feature(
            self,
            feature: NDArrayFloatT,
            corrupt_coeff: float,
            dist_name: str,
    ) -> NDArrayFloatT:
        """
Add a corrupted version of a feature based on a specified corruption coefficient.

    This method generates a new feature by combining the original feature with a 
    noise component drawn from a specified distribution. The amount of noise added 
    is controlled by the `corrupt_coeff`, which determines the balance between the 
    original feature and the noise.

    Args:
        feature (NDArrayFloatT): The original feature to be corrupted.
        corrupt_coeff (float): The coefficient that determines the proportion of the 
            original feature versus the noise. Should be between 0 and 1.
        dist_name (str): The name of the distribution to sample noise from. This 
            should correspond to a key in the `self.distribution` dictionary.

    Returns:
        NDArrayFloatT: The corrupted feature, which is a combination of the original 
        feature and the noise sampled from the specified distribution.
    """
        return (feature * corrupt_coeff
                + self.distribution[dist_name](size=feature.shape) * (1 - corrupt_coeff))

    @staticmethod
    def add_second_order_feature(
            feature_first: NDArrayFloatT,
            feature_second: NDArrayFloatT
    ) -> NDArrayFloatT:
        """
Calculates the second-order feature by multiplying two input features.

    This method takes two features as input and returns their element-wise 
    product, which represents a second-order feature.

    Args:
        feature_first (NDArrayFloatT): The first feature array.
        feature_second (NDArrayFloatT): The second feature array.

    Returns:
        NDArrayFloatT: The resulting array from the element-wise multiplication 
        of the two input features.
    """
        return feature_first * feature_second
