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

from ote_sdk.entities.id import ID
from ote_sdk.entities.label import Color
from ote_sdk.entities.label import Domain
from ote_sdk.entities.label import LabelEntity
from ote_sdk.utils.time_utils import now


class Label(LabelEntity):
    # pylint: disable=redefined-builtin, too-many-arguments; Requires refactor
    def __init__(
        self,
        name: str,
        domain: Domain,
        color: Optional[Color] = None,
        creation_date: Optional[datetime.datetime] = None,
        is_empty: bool = False,
        id: Optional[ID] = None,
    ):
        id = ID() if id is None else id
        color = Color.random() if color is None else color
        creation_date = now() if creation_date is None else creation_date
        super().__init__(
            name=name,
            id=id,
            is_empty=is_empty,
            color=color,
            domain=domain,
            creation_date=creation_date,
        )
