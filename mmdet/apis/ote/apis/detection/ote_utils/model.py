import datetime
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.metrics import NullPerformance
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model import ModelConfiguration
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model import ModelPrecision
from ote_sdk.entities.model import ModelStatus
from ote_sdk.entities.model_template import TargetDevice
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter
from ote_sdk.utils.time_utils import now


class Model(ModelEntity):
    def __init__(
        self,
        train_dataset: DatasetEntity,
        configuration: ModelConfiguration,
        *,
        creation_date: Optional[datetime.datetime] = None,
        performance: Optional[Performance] = None,
        previous_trained_revision: Optional[ModelEntity] = None,
        previous_revision: Optional[ModelEntity] = None,
        tags: Optional[List[str]] = None,
        data_source_dict: Optional[Mapping[str, Union["DataSource", bytes]]] = None,
        model_status: ModelStatus = ModelStatus.SUCCESS,
        training_duration: float = 0.0,
        precision: List[ModelPrecision] = None,
        latency: int = 0,
        fps_throughput: int = 0,
        target_device: TargetDevice = TargetDevice.CPU,
        target_device_type: Optional[str] = None,
        _id: Optional[ID] = None,
    ):
        _id = ID() if _id is None else _id
        performance = NullPerformance() if performance is None else performance
        creation_date = now() if creation_date is None else creation_date
        previous_trained_revision = (
            None
            if previous_trained_revision is None
            else previous_trained_revision
        )
        previous_revision = (
            None if previous_revision is None else previous_revision
        )

        try:
            version = previous_revision.version + 1
        except AttributeError:
            version = 1

        tags = [] if tags is None else tags
        precision = [] if precision is None else precision

        if not data_source_dict:
            if model_status == ModelStatus.SUCCESS:
                raise ValueError(
                    "A data_source_dict must be provided for a successfully trained model."
                )
            data_source_dict = {}
        model_adapters = {
            key: ModelAdapter(val) for key, val in data_source_dict.items()
        }

        super().__init__(
            _id=_id,
            creation_date=creation_date,
            train_dataset=train_dataset,
            previous_trained_revision=previous_trained_revision,
            previous_revision=previous_revision,
            version=version,
            tags=tags,
            model_status=model_status,
            performance=performance,
            training_duration=training_duration,
            configuration=configuration,
            model_adapters=model_adapters,
            precision=precision,
            latency=latency,
            fps_throughput=fps_throughput,
            target_device=target_device,
            target_device_type=target_device_type,
        )
