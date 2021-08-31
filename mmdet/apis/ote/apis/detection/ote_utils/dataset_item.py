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

import copy
import itertools
import logging
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.shapes.box import Box
from ote_sdk.entities.subset import Subset
from ote_sdk.utils.shape_factory import ShapeFactory

from .image import Image

logger = logging.getLogger(__name__)


# FIXME.
class AnnotationScene:
    def __init__(self, annotation):
        self.annotations = annotation


class MMDatasetItem(DatasetItemEntity):

    def __init__(self,
        image: Image,
        annotation: Sequence[Annotation] = (),
        subset: Subset = Subset.NONE
    ):
        self.image: Image = image
        self.annotation: List[Annotation] = list(annotation)
        # FIXME.
        self.annotation_scene = AnnotationScene(self.annotation)
        self.subset: Subset = subset
        self._roi = Annotation(Box.generate_full_box(), labels=[], id=ID())

    def __repr__(self):
        return f'MMDatasetItem({self.image}, {self.annotation}, {self.subset})'

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
        include_empty: bool = False,
        ios_threshold: float = 0.0,
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
        self.annotation.extend(validated_annotations)

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
