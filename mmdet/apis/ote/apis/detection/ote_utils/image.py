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
from typing import Tuple

import cv2
import numpy as np
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.id import ID
from ote_sdk.entities.media import IMedia2DEntity
from ote_sdk.entities.media_identifier import MediaIdentifierEntity
from ote_sdk.entities.shapes.box import Box
from ote_sdk.utils.time_utils import now


class ImageIdentifier(MediaIdentifierEntity):

    identifier_name = "image"

    def __init__(self, image_id: Optional[ID] = None):
        self.__media_id = image_id if image_id is not None else ID()

    @property
    def media_id(self) -> ID:
        return self.__media_id

    def __repr__(self):
        return f"ImageIdentifier(type={str(self.identifier_name)} media={str(self.media_id)})"

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
        name: str = '',
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

    def __repr__(self):
        s = ''
        if self.__data is not None:
            s += 'With data'
        else:
            s += f'file: "{self.__file_path}"'
        return f'Image({s}, {self.width}, {self.height}, {self.media_identifier})'

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
        if self.__data is None:
            self.__data = cv2.imread(self.__file_path)
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
