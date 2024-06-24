from typing import Tuple, TypeVar
import numpy as np
import numpy.typing as npt

ActivationType = TypeVar("ActivationType", npt.NDArray[np.float64], float)


def xavier_initialization(
    shape: Tuple[int, int],
) -> np.ndarray[Tuple[int, int], np.dtype[np.float64]]:
    """
    Xavier initialization for the weight matrix. Good for sigmoid and tanh activation function.
    Initialized using a normal distribution with mean = 0 and variance = 1 / input_size.

    Args:
        shape: Dimensions of the weight matrix (input_size, output_size).

    Returns:
        Initialized weight matrix.
    """
    # np.random.randn returns a sample (or samples) from the "standard normal" distribution.
    # Multiply the sample by the square root of 1 / input_size to get the desired variance.
    # Mean is the same as the standard normal distribution, 0.
    return np.random.randn(*shape) * np.sqrt(
        1 / shape[0]
    )  # Destructure the shape using `*`


def he_initialization(
    shape: Tuple[int, int],
) -> np.ndarray[Tuple[int, int], np.dtype[np.float64]]:
    """
    He initialization for the weight matrix. Good for ReLU activation function.
    Initialized using a normal distribution with mean = 0 and variance = 2 / input_size.

    Args:
        shape: Dimensions of the weight matrix (input_size, output_size).

    Returns:
        Initialized weight matrix.
    """
    # np.random.randn returns a sample (or samples) from the "standard normal" distribution.
    # Multiply the sample by the square root of 2 / input_size to get the desired variance.
    # Mean is the same as the standard normal distribution, 0.
    return np.random.randn(*shape) * np.sqrt(
        2 / shape[0]
    )  # Destructure the shape using `*`


def z(
    prev_layer: np.ndarray[Tuple[int], np.dtype[np.float64]],
    weights: np.ndarray[Tuple[int, int], np.dtype[np.float64]],
    biases: np.ndarray[Tuple[int], np.dtype[np.float64]],
):
    return np.dot(prev_layer, weights) + biases


def tanh(x: ActivationType) -> ActivationType:
    return np.tanh(x)  # type: ignore


def tanh_derivative(x: ActivationType) -> ActivationType:
    return 1 - np.tanh(x) ** 2  # type: ignore


def sigmoid(x: ActivationType) -> ActivationType:
    return 1 / (1 + np.exp(-x))  # type: ignore


def sigmoid_derivative(x: ActivationType) -> ActivationType:
    return x * (1 - x)  # type: ignore


def relu(x: ActivationType) -> ActivationType:
    return np.maximum(0, x)  # type: ignore


def relu_derivative(x: ActivationType) -> ActivationType:
    return np.where(x > 0, 1, 0)  # type: ignore
