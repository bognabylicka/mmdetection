# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import datetime
from typing import Optional

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.metrics import NullPerformance
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.resultset import ResultsetPurpose
from ote_sdk.utils.time_utils import now


class ResultSet(ResultSetEntity):
    # pylint: disable=redefined-builtin, too-many-arguments; Requires refactor
    def __init__(
        self,
        model: ModelEntity,
        ground_truth_dataset: DatasetEntity,
        prediction_dataset: DatasetEntity,
        purpose: ResultsetPurpose = ResultsetPurpose.EVALUATION,
        performance: Optional[Performance] = None,
        creation_date: Optional[datetime.datetime] = None,
        id: Optional[ID] = None,
    ):
        id = ID() if id is None else id
        performance = NullPerformance() if performance is None else performance
        creation_date = now() if creation_date is None else creation_date
        super().__init__(
            id=id,
            model=model,
            prediction_dataset=prediction_dataset,
            ground_truth_dataset=ground_truth_dataset,
            performance=performance,
            purpose=purpose,
            creation_date=creation_date,
        )
