"""
Reading/Loading data from the MNIST dataset

Classes:
    TrainingData: Loads and holds the training images and labels from the MNIST dataset.
    TestingData: Loads and holds the testing images and labels from the MNIST dataset.

Functions:
    read_labels(file_name: str) -> npt.NDArray[np.uint8]: Reads the labels from the dataset file.
    read_images(file_name: str) -> np.ndarray[Tuple[int, int, int], np.dtype[np.uint8]]:
    Reads the images from the dataset file.

References:
    - Yann LeCun's MNIST database: http://yann.lecun.com/exdb/mnist/
    - StackOverflow discussion on loading MNIST data: https://stackoverflow.com/a/53570674/23929926
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ModelData:
    """
    A class used to load and hold the training images and labels from the MNIST dataset.

    Attributes:
        images: A 3D numpy array containing the training images.
        labels: A 1D numpy array containing the training labels.
    """

    images: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]
    labels: np.ndarray[Tuple[int], np.dtype[np.uint8]]


def get_training_data() -> ModelData:
    """
    Used to load the training images and labels from the MNIST dataset.

    Returns:
        A ModelData object containing the training images and labels.
    """

    images = read_images("data/train-images.idx3-ubyte")
    labels = read_labels("data/train-labels.idx1-ubyte")
    return ModelData(images, labels)


def get_testing_data() -> ModelData:
    """
    Used to load the testing images and labels from the MNIST dataset.

    Returns:
        A ModelData object containing the testing images and labels.
    """

    images = read_images("data/test-images.idx3-ubyte")
    labels = read_labels("data/test-labels.idx1-ubyte")
    return ModelData(images, labels)


def read_labels(file_name: str) -> np.ndarray[Tuple[int], np.dtype[np.uint8]]:
    """
    Used to read the labels from the dataset file

    Args:
        file_name: The path to the `.idx1-ubyte` file containing the labels.

    Returns:
        A 1D numpy array containing the labels.
    """
    with open(file_name, "rb") as f:
        f.read(8)  # Skip the magic number and the number of labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)  # 1D array for all the labels
    return labels


def read_images(
    file_name: str,
) -> np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]:
    """
    Used to read the images from the dataset file

    Args:
        file_name: The path to the `.idx3-ubyte` file containing the images.

    Returns:
        A 3D numpy array containing the images with shape (num_images, num_rows, num_cols).
    """
    with open(file_name, "rb") as f:
        f.read(4)  # Skip the magic number
        num_images = int.from_bytes(f.read(4), "big")  # "big" for big-endian
        num_rows = int.from_bytes(f.read(4), "big")  # Dimension of the image
        num_cols = int.from_bytes(f.read(4), "big")  # Dimension of the image
        images = np.frombuffer(f.read(), dtype=np.float64).reshape(
            num_images, num_rows, num_cols
        )  # 3D array for all the images with dimensions 28x28
    return images
