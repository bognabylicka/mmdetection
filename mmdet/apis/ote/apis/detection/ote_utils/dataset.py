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

import collections.abc
import datetime
import json
import os
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union
from typing import overload

import numpy as np
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.annotation import AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.datasets import DatasetPurpose
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.shapes.box import Box
from ote_sdk.entities.subset import Subset
from ote_sdk.utils.time_utils import now

from mmdet.datasets import CocoDataset

from .dataset_item import MMDatasetItem
from .image import Image
from .label import Label


class DatasetIterator(collections.abc.Iterator):
    """
    This DatasetIterator iterates over the dataset lazily.
    Implements collections.abc.Iterator.
    """

    def __init__(self, dataset: DatasetEntity):
        self.dataset = dataset
        self.index = 0

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        item = self.dataset[self.index]
        self.index += 1
        return item


class MMDataset(DatasetEntity, Iterable[DatasetItemEntity]):

    def __init__(
        self,
        annotation_files: Optional[Mapping[Subset, str]] = None,
        data_root_dirs: Optional[Mapping[Subset, str]] = None,
        items: Optional[List[MMDatasetItem]] = None,
        labels: List[LabelEntity] = (),
        creation_date: Optional[datetime.datetime] = None,
        id: Optional[ID] = None,
        purpose: DatasetPurpose = DatasetPurpose.INFERENCE
    ):
        id = id if id is not None else ID()
        creation_date = creation_date if creation_date is not None else now()
        super().__init__(id=id, creation_date=creation_date, items=None, purpose=purpose, mutable=False)

        self.labels = list(labels)

        self._annotation_files = annotation_files
        self._data_root_dirs = data_root_dirs
        if items is not None:
            self._items = items
        else:
            ann_files = {}
            for k, v in annotation_files.items():
                if v:
                    ann_files[k] = os.path.abspath(v)

            data_roots = {}
            for k, v in data_root_dirs.items():
                if v:
                    data_roots[k] = os.path.abspath(v)

            for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
                self.add_data(ann_files.get(subset), data_roots.get(subset), subset)

    def _find_label_by_name(self, name):
        matching_labels = [label for label in self.labels if label.name == name]
        if len(matching_labels) == 1:
            return matching_labels[0]
        elif len(matching_labels) == 0:
            label = Label(name=name, domain="detection", id=len(self.labels))
            self.labels.append(label)
            return label
        else:
            raise ValueError('Found multiple matching labels')

    def add_data(self, ann_file: str, data_root: str, subset: Subset):
        test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
        if ann_file is None:
            return
        pipeline = [
            # dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
            ]
        classes = None
        with open(ann_file) as f:
            content = json.load(f)
            classes = [v['name'] for v in sorted(content['categories'], key=lambda x: x['id'])]
        coco_dataset = CocoDataset(ann_file=ann_file,
                                   pipeline=pipeline,
                                   data_root=data_root,
                                   classes=classes,
                                   test_mode=test_mode)
        coco_dataset.test_mode = False
        for label_name in classes:
            self._find_label_by_name(label_name)

        for item in coco_dataset:
            def create_gt_box(x1, y1, x2, y2, label_name):
                return Annotation(Box(x1=x1, y1=y1, x2=x2, y2=y2),
                                  labels=[ScoredLabel(label=self._find_label_by_name(label_name))])

            img_height = item['img_info'].get('height')
            img_width = item['img_info'].get('width')
            divisor = np.array([img_width, img_height, img_width, img_height], dtype=item['gt_bboxes'].dtype)
            bboxes = item['gt_bboxes'] / divisor
            labels = item['gt_labels']

            if item['img_prefix'] is not None:
                filename = os.path.join(item['img_prefix'], item['img_info']['filename'])
            else:
                filename = item['img_info']['filename']

            shapes = [create_gt_box(*coords, coco_dataset.CLASSES[label_id]) for coords, label_id in zip(bboxes, labels)]

            dataset_item = MMDatasetItem(Image(file_path=filename, name=filename), shapes, subset)
            self._items.append(dataset_item)

    def __repr__(self):
        s = f'{self.__class__.__name__}(id={self.id}, creation_date={self.creation_date}, purpose={self.purpose}, ' \
            f'labels={self.labels}, '
        if self._annotation_files is not None:
            s += f'annotation_files={self._annotation_files}, data_root_dirs={self._data_root_dirs}, '
        s = f'{len(self._items)} items'
        return s

    def __len__(self) -> int:
        return len(self._items)

    def get_subset(self, subset: Subset) -> "MMDataset":
        dataset = MMDataset(items=[x for x in self._items if x.subset == subset], purpose=self.purpose)
        return dataset

    def delete_items_with_media_id(self, media_id: ID):
        """
        Removes dataset items which are associated with media_id.

        :param media_id: the id of the media that will be removed
        """
        indices_to_delete = []
        for i, item in enumerate(self._items):
            if item.image.media_identifier.media_id == media_id:
                indices_to_delete.append(i)

        self.remove_at_indices(indices_to_delete)

    def __add__(self, other: Union["MMDataset", List[MMDatasetItem]]) -> "MMDataset":
        """
        Returns a new dataset which contains the items of self added with the input dataset.
        Note that additional info of the dataset might be incoherent to the addition operands.

        :param other: dataset to be added to output

        :return: new dataset
        """
        items: List[MMDatasetItem]
        if isinstance(other, MMDataset):
            items = self.__items + list(other)
        elif isinstance(other, list):
            items = self.__items + [o for o in other if isinstance(o, MMDatasetItem)]
        else:
            raise ValueError("Cannot add other than MMDataset entities")
        return MMDataset(items=items, purpose=self.purpose)

    def append(self, item: MMDatasetItem):
        """
        Append a DatasetItem to the dataset

        :param item: item to append
        """
        if item.image is None:
            raise ValueError("Image in dataset item cannot be None")
        self._items.append(item)

    def remove(self, item: MMDatasetItem):
        """
        Remove an item from the items.
        This function calls remove_at_indices function.

        :raises ValueError: if the input item is not in the dataset
        :param item: the item to be deleted
        """
        index = self._items.index(item)
        self.remove_at_indices([index])

    def remove_at_indices(self, indices: List[int]):
        """
        Delete items based on the `indices`.

        :param indices: the indices of the items that will be deleted from the items.
        """
        indices.sort(reverse=True)  # sort in descending order
        for i in indices:
            del self._items[i]

    def __eq__(self, other) -> bool:
        if (
            isinstance(other, DatasetEntity)
            and len(self) == len(other)
            and self.purpose == other.purpose
        ):
            if other.id != ID() and self.id != ID():
                return self.id == other.id
            return False not in [self[i] == other[i] for i in range(len(self))]

        return False

    def _fetch(self, key):
        if isinstance(key, list):
            return [self._fetch(ii) for ii in key]
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self._fetch(ii) for ii in range(*key.indices(len(self._items)))]
        if isinstance(key, (int, np.int32, np.int64)):
            return self._items[key]
        raise TypeError(
            f"Instance of type `{type(key).__name__}` cannot be used to access Dataset items. "
            f"Only slice and int are supported"
        )

    @overload
    def __getitem__(self, key: int) -> MMDatasetItem:
        return self._fetch(key)

    @overload  # Overload for proper type hinting of indexing on dataset
    def __getitem__(self, key: slice) -> List[MMDatasetItem]:
        return self._fetch(key)

    def __getitem__(self, key):
        return self._fetch(key)

    def __iter__(self) -> Iterator[MMDatasetItem]:
        return DatasetIterator(self)

    def sort_items(self):
        self._items = sorted(
            self._items, key=lambda x: (x.image.media_identifier.media_id)
        )

    def contains_media_id(self, media_id: ID) -> bool:
        """
        Returns True if there are any dataset items which are associated with media_id.

        :param media_id: the id of the media that is being checked.

        :return: True if there are any dataset items which are associated with media_id.
        """
        for item in self:
            if item.image.media_identifier.media_id == media_id:
                return True
        return False

    def with_empty_annotations(
        self, annotation_kind: AnnotationSceneKind = AnnotationSceneKind.PREDICTION
    ) -> "MMDataset":
        new_items = [MMDatasetItem(x.image, annotation=[], subset=x.subset) for x in self._items]
        new_dataset = MMDataset(items=new_items, purpose=self.purpose)
        return new_dataset

    def get_labels(self, include_empty: bool = False) -> List[Label]:
        """
        Returns the list of all unique labels that are in the dataset, this does not respect the ROI of the dataset
        items.

        :param include_empty: set to True to include empty label (if exists) in the output.
        :return: list of labels that appear in the dataset
        """
        return self.labels
        # label_set = set(
        #     itertools.chain(
        #         *[item.annotation.get_labels(include_empty) for item in iter(self)]
        #     )
        # )
        # return list(label_set)
