# This import fixes the problem that specifying the type of an object
# in its module definition raises error
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Iterator

import numpy as np

from app.utilities.img import Image

# the vector representing a crossed box
crossed_vec = np.array([[0], [1]])
# the vector representing a empty box
empty_vec = np.array([[1], [0]])

"""The following two folder names should be set before starting the training"""
# the default folder name where the crossed boxes locate
_default_training_data_crossed_box_folder_name = ''
# the default folder name where the empty boxes locate
_default_training_data_empty_box_folder_name = ''


class TrainingBatchDistributor:
    """This class is an interface to the training data

    Call 'iter_next_training_batch' to iterate through the training data,
    including an input vector transformed from the training image and an expectation vector

    Call 'iter_rest' to iterate the rest of the data for verification
    """

    # the training image data extension
    _training_data_extension = '.png'
    # the proportion of the data used to train the dnn and the data used to verify the correctness
    # For example 0.75 means 75% of the data will be used to train the dnn
    # the rest 25% will be used to verify its correctness
    _training_proportion: float = 0.75

    def __init__(self):
        """Initializes the distributor"""
        crossed_path = Path(_default_training_data_crossed_box_folder_name)
        empty_path = Path(_default_training_data_empty_box_folder_name)

        def get_image_paths(folder_path: Path) -> List[Path]:
            return [p for p in folder_path.iterdir() if p.suffix == self._training_data_extension]

        image_paths_list = get_image_paths(crossed_path) + get_image_paths(empty_path)
        random.shuffle(image_paths_list)

        data_size = len(image_paths_list)
        splitting_index = round(data_size * self._training_proportion)

        self._training_data_paths = image_paths_list[:splitting_index]
        self._verification_data_paths = image_paths_list[splitting_index:]

    def iter_next_training_batch(self, batch_size: int) -> Iterator[TrainingData]:
        """Returns Iterator to iterate through the training data

        including:
        -- an input vector transformed from the training image and an expectation vector
        -- an expectation vector

        Parameters
        ----------
        batch_size:
            the size of a batch

        Returns
        -------
        iterator:
            Iterator to iterate through the training data
        """
        if len(self._training_data_paths) < batch_size:
            raise NoMoreTrainingData("Training data are not sufficient to form a new training batch!")
        else:
            sets = [TrainingData(self._training_data_paths.pop()) for i in range(0, batch_size)]
            return sets.__iter__()

    def iter_rest(self) -> Iterator[TrainingData]:
        """Iterate the rest data to verify the correctness of the dnn"""
        sets_rest = [TrainingData(p) for p in self._verification_data_paths]
        return sets_rest.__iter__()


class NoMoreTrainingData(Exception):
    """Raised when there is not sufficient training data to form a batch"""
    pass


class TrainingData:
    # the default 1 Byte gray scale range size
    _gray_scale_range_size = 256

    def __init__(self, path_to_image: Path):
        """Initializes the training data

        Parameters
        ----------
        path_to_image:
            the path object to the training image data
        """
        # convert the image to gray scale then to data array
        image_data = Image.open_extern(path_to_image).to_gray_scale().to_data_ndarray()
        # reshape and normalize the input
        self._input = image_data.reshape(image_data.size, 1) / (self._gray_scale_range_size - 1)
        # find the expectation by analysing the image path
        if path_to_image.parent.resolve() == Path(_default_training_data_crossed_box_folder_name).resolve():
            self._expectation = crossed_vec
        else:
            self._expectation = empty_vec

    @property
    def input(self) -> np.ndarray:
        """Gets the input vector"""
        return self._input

    @property
    def expectation(self) -> np.ndarray:
        """Gets the expectation"""
        return self._expectation
