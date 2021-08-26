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
import copy
import datetime
import itertools
import json
import logging
import os
from copy import deepcopy
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
from typing import overload

import cv2
import numpy as np
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.annotation import AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.datasets import DatasetPurpose
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.media import IMedia2DEntity
from ote_sdk.entities.media_identifier import MediaIdentifierEntity
from ote_sdk.entities.shapes.box import Box
from ote_sdk.entities.subset import Subset
from ote_sdk.utils.shape_factory import ShapeFactory
from ote_sdk.utils.time_utils import now

from mmdet.datasets import CocoDataset
from mmdet.datasets import ConcatDataset
from mmdet.datasets import ConcatenatedCocoDataset

# from sc_sdk.entities.annotation import AnnotationScene
# from sc_sdk.entities.annotation import NullMediaIdentifier
# from sc_sdk.entities.image import Image
# from sc_sdk.entities.interfaces.dataset_adapter_interface import DatasetAdapterInterface


logger = logging.getLogger(__name__)

class ImageIdentifier(MediaIdentifierEntity):

    identifier_name = "image"

    def __init__(self, image_id: Optional[ID] = None):
        self.__media_id = image_id if image_id is None else ID()

    @property
    def media_id(self) -> ID:
        return self.__media_id

    def __repr__(self):
        return f"ImageIdentifier(type={str(self.identifier_name)} media={str(self.media_id)})"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if isinstance(other, ImageIdentifier):
            return self.media_id == other.media_id
        return False

    def __hash__(self):
        return hash(str(self))

    def as_tuple(self) -> tuple:
        return self.identifier_name, self.__media_id


class Image(IMedia2DEntity):

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        file_path: Optional[str] = None,
        name: Optional[str] = '',
        creation_date: Optional[datetime.datetime] = None,
        image_id: Optional[ImageIdentifier] = None
    ):
        creation_date = creation_date if creation_date is not None else now()
        super().__init__(name, creation_date)
        self.__data = data
        self.__file_path = file_path
        self.__height = None
        self.__width = None
        self.__image_id = image_id if image_id is not None else ImageIdentifier()

    def __get_size(self) -> Tuple[int, int]:
        if self.__data is not None:
            return self.__data.shape[:2]
        # TODO. Get image size w/o reading & decoding its data.
        image = cv2.imread(self.__file_path)
        return image.shape[:2]

    @property
    def media_identifier(self) -> MediaIdentifierEntity:
        return self.__image_id

    @property
    def numpy(self) -> np.ndarray:
        return self.__data

    def roi_numpy(self, roi: Optional[Annotation] = None) -> np.ndarray:
        """
        Obtains the numpy representation of the image for a selection region of interest (roi).

        :param roi: The region of interest can be any shape in the relative coordinate system of the full-annotation.
        :return: selected region as numpy
        """
        data = self.numpy
        if roi is None:
            return data

        if not isinstance(roi.shape, Box):
            raise ValueError("roi shape is not a Box")

        if data is None:
            raise ValueError("Numpy array is None, and thus cannot be cropped")

        if len(data.shape) < 2:
            raise ValueError(
                "This image is one dimensional, and thus cannot be cropped"
            )

        return roi.shape.crop_numpy_array(data)

    @property
    def height(self) -> int:
        if self.__height is None:
            self.__height, self.__width = self.__get_size()
        return self.__height

    @property
    def width(self) -> int:
        if self.__width is None:
            self.__height, self.__width = self.__get_size()
        return self.__width


class MMDatasetItem(DatasetItemEntity):

    def __init__(self,
        image: Image,
        annotation: Optional[Sequence[Annotation]] = (),
        subset: Subset = Subset.NONE
    ):
        self.image: Image = image
        self.annotation: List[Annotation] = list(annotation)
        self.subset: Subset = subset
        self._roi = Annotation(Box.generate_full_box(), labels=[], id=ID())

    @property
    def roi(self) -> Annotation:
        return self._roi

    @roi.setter
    def roi(self, roi: Annotation):
        # Disable ROI change. It's always a full box.
        raise NotImplementedError

    def roi_numpy(self, roi: Optional[Annotation] = None) -> np.ndarray:
        return self.image.roi_numpy(roi)

    @property
    def numpy(self) -> np.ndarray:
        return self.image.numpy

    def get_roi_labels(
        self, labels: Optional[List[LabelEntity]] = None, include_empty: bool = False
    ) -> List[LabelEntity]:
        """
        Return the subset of the input labels which exist in the dataset item (wrt. ROI).

        :param labels: Subset of input labels to filter with; if ``None``, all the labels within the ROI are returned
        :param include_empty: if True, returns both empty and non-empty labels
        :return: The intersection of the input label set and those present within the ROI
        """
        filtered_labels = set()
        for label in self.roi.get_labels(include_empty):
            if labels is None or label.get_label() in labels:
                filtered_labels.add(label.get_label())
        return sorted(list(filtered_labels), key=lambda x: x.name)

    def get_annotations(
        self,
        labels: Optional[List[LabelEntity]] = None,
        include_empty: Optional[bool] = False,
        ios_threshold: Optional[float] = 0.0,
    ) -> List[Annotation]:
        """
        Returns a list of annotations that exist in the dataset item (wrt. ROI)

        :param labels: Subset of input labels to filter with; if ``None``, all the shapes within the ROI are returned
        :param include_empty: if True, returns both empty and non-empty labels
        :param ios_threshold: Only return shapes where Area(self.roi ∩ shape)/ Area(shape) > ios_threshold.
        :return: The intersection of the input label set and those present within the ROI
        """
        annotations = []
        if labels is None and not include_empty:
            # Fast path for the case where we do not need to change the shapes
            annotations = self.annotation
        else:
            labels_set = {label.name for label in labels} if labels is not None else {}
            for annotation in self.annotation:
                shape_labels = annotation.get_labels(include_empty)
                if labels is not None:
                    shape_labels = [
                        label for label in shape_labels if label.name in labels_set
                    ]
                    if len(shape_labels) == 0:
                        continue
                # Also create a copy of the shape, so that we can safely modify the labels
                # without tampering with the original shape.
                shape = copy.deepcopy(annotation.shape)
                annotations.append(Annotation(shape=shape, labels=shape_labels))
        return annotations

    def get_shapes_labels(
        self, labels: List[LabelEntity] = None, include_empty: bool = False
    ) -> List[LabelEntity]:
        """
        Get the labels of the shapes present in this dataset item. if a label list is supplied, only labels present
        within that list are returned. if include_empty is True, present empty labels are returned as well.

        :param labels: if supplied only labels present in this list are returned
        :param include_empty: if True, returns both empty and non-empty labels
        :return: a list of labels from the shapes within the roi of this dataset item
        """
        annotations = self.get_annotations()
        scored_label_set = set(
            itertools.chain(
                *[annotation.get_labels(include_empty) for annotation in annotations]
            )
        )
        label_set = {scored_label.get_label() for scored_label in scored_label_set}

        if labels is None:
            return list(label_set)
        return [label for label in label_set if label in labels]


    def append_annotations(self, annotations: Sequence[Annotation]):
        """
        Adds a list of shapes to the annotation
        """
        validated_annotations = [
            annotation
            for annotation in annotations
            if ShapeFactory().shape_produces_valid_crop(
                shape=annotation.shape,
                media_width=self.image.width,
                media_height=self.image.height,
            )
        ]
        n_invalid_shapes = len(annotations) - len(validated_annotations)
        if n_invalid_shapes > 0:
            logger.info(
                "%d shapes will not be added to the dataset item as they "
                "would produce invalid crops (this is expected for some tasks, "
                "such as segmentation).",
                n_invalid_shapes,
            )
        self.annotation.append(validated_annotations)

    def append_labels(self, labels: List[ScoredLabel]):
        """
        Appends labels to the DatasetItem and adds it to the the annotation label as well if it's not yet there

        :param labels: list of labels to be appended
        """
        # TODO. What's going on here?
        if len(labels) == 0:
            return

        annotation = None
        if annotation is None:  # no annotation found with shape
            annotation = self.roi
            self.annotation.append(annotation)

        for label in labels:
            if label not in self.roi.get_labels(include_empty=True):
                self.roi.append_label(label)
            if label not in annotation.get_labels(include_empty=True):
                annotation.append_label(label)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height



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

    def __init__(self,
                 annotation_files: Optional[Mapping[Subset, str]] = None,
                 data_root_dirs: Optional[Mapping[Subset, str]] = None,
                 subsets: Optional[List[CocoDataset]] = None,
                 creation_date: Optional[datetime.datetime] = None,
                 id: Optional[ID] = None,
                 purpose: Optional[DatasetPurpose] = DatasetPurpose.INFERENCE
                 ):
        id = id if id is not None else ID()
        creation_date = creation_date if creation_date is not None else now()
        super().__init__(id=id, creation_date=creation_date, items=None, purpose=purpose, mutable=False)

        if subsets is not None:
            self.subsets = subsets
        else:
            ann_files = {}
            for k, v in annotation_files.items():
                if v:
                    ann_files[k] = os.path.abspath(v)

            data_roots = {}
            for k, v in data_root_dirs.items():
                if v:
                    data_roots[k] = os.path.abspath(v)

            self.subsets = {}
            non_empty_subsets = []
            for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
                subdataset = self.__init_subset(ann_files.get(subset), data_roots.get(subset), subset)
                self.subsets[subset] = subdataset
                if subdataset is not None:
                    non_empty_subsets.append(subdataset)

            if len(non_empty_subsets) == 0:
                raise ValueError('Dataset is empty.')

            self.subsets[Subset.NONE] = ConcatenatedCocoDataset(ConcatDataset(non_empty_subsets))
            self.subsets[Subset.NONE].copy_paste_aug_used = False

        self.coco_dataset = self.subsets[Subset.NONE]
        self.project_labels = None

    # FIXME
    def set_project_labels(self, project_labels):
        self.project_labels = project_labels

    def label_name_to_project_label(self, label_name):
        return [label for label in self.project_labels if label.name == label_name][0]

    def __init_subset(self, ann_file, data_root, subset) -> bool:
        test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
        if ann_file is None:
            return None
        pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True)]
        classes = None
        with open(ann_file) as f:
            content = json.load(f)
            classes = [v['name'] for v in sorted(content['categories'], key=lambda x: x['id'])]
        coco_dataset = CocoDataset(ann_file=ann_file,
                                   pipeline=pipeline,
                                   data_root=data_root,
                                   classes=classes,
                                   test_mode=test_mode)
        if hasattr(coco_dataset, 'flag'):
            del coco_dataset.flag
        coco_dataset.test_mode = False
        return coco_dataset

    def __repr__(self):
        return f"MMDataset(\n" \
            f"\ttrain_ann_file={self.ann_files[Subset.TRAINING]},\n" \
            f"\ttrain_data_root={self.data_roots[Subset.TRAINING]},\n" \
            f"\tval_ann_file={self.ann_files[Subset.VALIDATION]},\n" \
            f"\tval_data_root={self.data_roots[Subset.VALIDATION]},\n" \
            f"\ttest_ann_file={self.ann_files[Subset.TESTING]},\n" \
            f"\ttest_data_root={self.data_roots[Subset.TESTING]},\n" \
            f"\tsize={len(self)})"

    def __len__(self) -> int:
        assert self.coco_dataset is not None
        return len(self.coco_dataset)

    def get_subset(self, subset: Subset) -> "MMDataset":
        subdatasets = {
            subset: self.subsets[subset],
            Subset.NONE: self.subsets[subset]
        }
        dataset = MMDataset(subsets=subdatasets, id=self.id, creation_date=self.creation_date, purpose=self.purpose)
        dataset.project_labels = self.project_labels
        return dataset

    # Disable methods that modify the dataset, treating it as immutable.

    def delete_items_with_media_id(self, media_id: ID):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def append(self, item):
        raise NotImplementedError

    def remove(self, item):
        raise NotImplementedError

    def remove_at_indices(self, indices):
        raise NotImplementedError

    ############################################################

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

    ############################################################

    def _fetch(self, key):
        if isinstance(key, list):
            return [self._fetch(ii) for ii in key]
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self._fetch(ii) for ii in range(*key.indices(len(self._items)))]
        if isinstance(key, (int, np.int32, np.int64)):
            def create_gt_scored_label(label_name):
                return ScoredLabel(label=self.label_name_to_project_label(label_name))

            def create_gt_box(x1, y1, x2, y2, label):
                return Annotation(Box(x1=x1, y1=y1, x2=x2, y2=y2),
                                  labels=[create_gt_scored_label(label)])

            item = self.coco_dataset[key]
            divisor = np.tile([item['ori_shape'][:2][::-1]], 2)
            bboxes = item['gt_bboxes'] / divisor
            labels = item['gt_labels']

            shapes = [create_gt_box(*coords, self.subsets[Subset.NONE].CLASSES[label_id]) for coords, label_id in zip(bboxes, labels)]

            # FIXME. Subset.NONE is not correct.
            dataset_item = MMDatasetItem(Image(data=item['img']), shapes, Subset.NONE)

            # # FIXME. Support lazy data fetching.
            # image = Image(data=item['img'])
            # annotation_scene = AnnotationScene(kind=AnnotationSceneKind.ANNOTATION,
            #                                 media_identifier=NullMediaIdentifier(),
            #                                 annotations=shapes)
            # dataset_item = DatasetItem(image, annotation_scene)
            return dataset_item
        raise TypeError(
            f"Instance of type `{type(key).__name__}` cannot be used to access Dataset items. "
            f"Only slice and int are supported"
        )

    @overload
    def __getitem__(self, key: int) -> DatasetItemEntity:
        return self._fetch(key)

    @overload  # Overload for proper type hinting of indexing on dataset
    def __getitem__(self, key: slice) -> List[DatasetItemEntity]:
        return self._fetch(key)

    def __getitem__(self, key):
        return self._fetch(key)

    def __iter__(self) -> Iterator[DatasetItemEntity]:
        return DatasetIterator(self)

    def sort_items(self):
        raise NotImplementedError

    def contains_media_id(self, media_id: ID) -> bool:
        """
        Returns True if there are any dataset items which are associated with media_id.

        :param media_id: the id of the media that is being checked.

        :return: True if there are any dataset items which are associated with media_id.
        """
        for item in iter(self):
            # FIXME.
            if item.media_identifier.media_id == media_id:
                return True
        return False

    ############################################################

    # @abc.abstractmethod
    # def get_labels(self, include_empty: bool = False) -> List[LabelEntity]:
    #     """
    #     Returns the list of all unique labels that are in the dataset, this does not
    #     respect the ROI of the dataset items.

    #     :param include_empty: set to True to include empty label (if exists) in the
    #            output.
    #     :return: list of labels that appear in the dataset
    #     """
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def with_empty_annotations(
    #     self, annotation_kind: AnnotationSceneKind = AnnotationSceneKind.PREDICTION
    # ):
    #     """
    #     Produces a new dataset with empty annotation objects (no shapes or labels).

    #     :return: a new dataset containing the same items, with empty annotation objects.
    #     """
    #     raise NotImplementedError

    ############################################################

    # def __copy__(self):
    #     """
    #     Shallow copy the dataset entity

    #     :return: The copied dataset
    #     """
    #     return Dataset(
    #         id=self.id,
    #         dataset_storage=self.dataset_storage,
    #         items=self.__items.copy(),
    #         purpose=self.purpose,
    #         mutable=self.mutable,
    #     )

    def with_empty_annotations(
        self, annotation_kind: AnnotationSceneKind = AnnotationSceneKind.PREDICTION
    ) -> "MMDataset":
        """
        Produces a new dataset with empty annotation objects (no shapes or labels).
        This is a convenience function to generate a dataset with empty annotations from another dataset.
        This is particularly useful for evaluation on validation data and to build resultsets.

        Assume a dataset containing user annotations.

        >>> labeled_dataset = Dataset()  # user annotated dataset

        Then, we want to see the performance of our task on this labeled_dataset,
        which means we need to create a new dataset to be passed for analysis.

        >>> prediction_dataset = labeled_dataset.with_empty_annotations()

        Later, we can pass this prediction_dataset to the task analysis function.
        By pairing the labeled_dataset and the prediction_dataset, the resultset can then be constructed.
        Refer to :class:`~sc_sdk.entities.resultset.ResultSet` for more info.

        :param annotation_kind: Sets the empty annotation to this kind. Default value: AnnotationSceneKind.PREDICTION
        :return: a new dataset containing the same items, with empty annotation objects.
        """
        new_dataset = MMDataset(subsets=self.subsets)
        new_dataset.project_labels = self.project_labels

        for key in range(len(self)):
            gt_boxes = new_dataset.coco_dataset[key]['gt_bboxes']
            shape = list(gt_boxes.shape)
            shape[0] = 0
            new_dataset.coco_dataset[key]['gt_bboxes'] = np.empty(shape, dtype=gt_boxes.dtype)
            gt_labels = new_dataset.coco_dataset[key]['gt_labels']
            new_dataset.coco_dataset[key]['gt_labels'] = np.empty(0, dtype=gt_labels.dtype)

        # for dataset_item in self:
        #     if isinstance(dataset_item, MMDatasetItem):
        #         new_dataset_item = MMDatasetItem(
        #             image=dataset_item.image,
        #             annotation=[],
        #             subset=dataset_item.subset,
        #         )
        #         new_dataset.append(new_dataset_item)
        return new_dataset


    def get_labels(self, include_empty: bool = False) -> List[LabelEntity]:
        """
        Returns the list of all unique labels that are in the dataset, this does not respect the ROI of the dataset
        items.

        :param include_empty: set to True to include empty label (if exists) in the output.
        :return: list of labels that appear in the dataset
        """
        label_set = set(
            itertools.chain(
                *[item.annotation_scene.get_labels(include_empty) for item in iter(self)]
            )
        )
        return list(label_set)




# class MMDataset(DatasetEntity):
#     def __init__(self,
#                  id: Optional[ID] = None,
#                  creation_date: Optional[datetime.datetime] = None,
#                  train_ann_file: Optional[str] = None,
#                  train_data_root: Optional[str] = None,
#                  val_ann_file: Optional[str] = None,
#                  val_data_root: Optional[str] = None,
#                  test_ann_file: Optional[str] = None,
#                  test_data_root: Optional[str] = None,
#                  **kwargs):
#         id = id if id is not None else ID()
#         creation_date = creation_date if creation_date is not None else now()
#         super().__init__(id, creation_date, **kwargs)
#         self.ann_files = {}
#         self.data_roots = {}
#         self.ann_files[Subset.TRAINING] = train_ann_file
#         self.data_roots[Subset.TRAINING] = train_data_root
#         self.ann_files[Subset.VALIDATION] = val_ann_file
#         self.data_roots[Subset.VALIDATION] = val_data_root
#         self.ann_files[Subset.TESTING] = test_ann_file
#         self.data_roots[Subset.TESTING] = test_data_root
#         self.coco_dataset = None
#         for k, v in self.ann_files.items():
#             if v:
#                 self.ann_files[k] = os.path.abspath(v)
#         for k, v in self.data_roots.items():
#             if v:
#                 self.data_roots[k] = os.path.abspath(v)
#         self.labels = None
#         self.set_labels_obtained_from_annotation()
#         self.project_labels = None

#     def set_labels_obtained_from_annotation(self):
#         self.labels = None
#         for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
#             path = self.ann_files[subset]
#             if path:
#                 labels = get_classes_from_annotation(path)
#                 if self.labels and self.labels != labels:
#                     raise RuntimeError('Labels are different from annotation file to annotation file.')
#                 self.labels = labels
#         assert self.labels is not None

#     def set_project_labels(self, project_labels):
#         self.project_labels = project_labels

#     def label_name_to_project_label(self, label_name):
#         return [label for label in self.project_labels if label.name == label_name][0]

#     def __getitem__(self, indx) -> dict:
#         def create_gt_scored_label(label_name):
#             return ScoredLabel(label=self.label_name_to_project_label(label_name))

#         def create_gt_box(x1, y1, x2, y2, label):
#             return Annotation(Box(x1=x1, y1=y1, x2=x2, y2=y2),
#                               labels=[create_gt_scored_label(label)])

#         item = self.coco_dataset[indx]
#         divisor = np.tile([item['ori_shape'][:2][::-1]], 2)
#         bboxes = item['gt_bboxes'] / divisor
#         labels = item['gt_labels']

#         shapes = [create_gt_box(*coords, self.labels[label_id]) for coords, label_id in zip(bboxes, labels)]

#         image = Image(name=None, numpy=item['img'], dataset_storage=NullDatasetStorage())
#         annotation_scene = AnnotationScene(kind=AnnotationSceneKind.ANNOTATION,
#                                            media_identifier=NullMediaIdentifier(),
#                                            annotations=shapes)
#         datset_item = DatasetItem(image, annotation_scene)
#         return datset_item

#     def __len__(self) -> int:
#         assert self.coco_dataset is not None
#         return len(self.coco_dataset)

#     def get_labels(self) -> list:
#         return self.labels

#     def init_as_subset(self, subset: Subset):
#         test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
#         if self.ann_files[subset] is None:
#             return False
#         pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True)]
#         self.coco_dataset = CocoDataset(ann_file=self.ann_files[subset],
#                                         pipeline=pipeline,
#                                         data_root=self.data_roots[subset],
#                                         classes=self.labels,
#                                         test_mode=test_mode)
#         self.coco_dataset.test_mode = False
#         return True

#     def get_subset(self, subset: Subset) -> Dataset:
#         dataset = deepcopy(self)
#         if dataset.init_as_subset(subset):
#             return dataset
#         return NullDataset()
