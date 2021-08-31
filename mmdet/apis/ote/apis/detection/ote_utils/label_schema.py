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

from typing import List
from typing import Optional

from ote_sdk.entities.label_schema import LabelGroup
from ote_sdk.entities.label_schema import LabelSchemaEntity

from .label import Label


class FlatLabelSchema(LabelSchemaEntity):

    def __init__(self, labels: List[Label]):
        self.labels = labels

    def get_labels(self, include_empty: bool) -> List[Label]:
        labels = [label for label in self.labels if include_empty or not label.is_empty]
        labels = sorted(labels, key=lambda x: x.id)
        return labels

    def get_groups(self, include_empty: bool = False) -> List[LabelGroup]:
        raise NotImplementedError

    def add_group(self, label_group: LabelGroup, exclusive_with: Optional[List[LabelGroup]] = None):
        raise NotImplementedError

    def add_child(self, parent: Label, child: Label):
        raise NotImplementedError

    def get_parent(self, label: Label) -> Optional[Label]:
        raise NotImplementedError
