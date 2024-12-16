from abc import ABC


class BaseSource(ABC):
    name: str

    def __init__(
            self,
            target_type: str | None,
            force_cat_features: list[str] | None,
            force_num_features: list[str] | None,
            drop_features: list[str] | None,
            **kwargs
    ) -> None:
        self.target_type = target_type
        self.force_cat_features = force_cat_features
        self.force_num_features = force_num_features
        self.drop_features = drop_features


class OpenMLSource(BaseSource):
    name = 'openml_source'

    def __init__(
            self,
            task_id: int,
            dataset_id: int | None = None,
            dataset_name: str | None = None,
            target_type: str | None = None,
            force_cat_features: list[str] | None = None,
            force_num_features: list[str] | None = None,
            drop_features: list[str] | None = None,
    ) -> None:
        super().__init__(
            target_type=target_type,
            force_cat_features=force_cat_features,
            force_num_features=force_num_features,
            drop_features=drop_features
        )
        self.task_id = task_id
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
