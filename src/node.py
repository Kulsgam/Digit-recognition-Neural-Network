"""
Provides various utility functions and initializers for neural networks,
including activation functions and weight initialization methods.

Functions:
    xavier_initialization: Initializes weights using Xavier initialization.
                           Good for sigmoid and tanh activation functions.
    he_initialization: Initializes weights using He initialization.
                       Good for ReLU activation functions.
    z: Computes the linear transformation of the input layer.
    tanh: Applies the hyperbolic tangent activation function.
    tanh_derivative: Computes the derivative of the hyperbolic tangent activation function.
    sigmoid: Applies the sigmoid activation function.
    sigmoid_derivative: Computes the derivative of the sigmoid activation function.
    relu: Applies the ReLU activation function.
    relu_derivative: Computes the derivative of the ReLU activation function.
"""

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
    """
    Computes the linear transformation of the input layer.

    Args:
        prev_layer: Activations from the previous layer.
        weights: Weights for the current layer.
        biases: Biases for the current layer.

    Returns:
        The result of the linear transformation.
    """
    return np.dot(prev_layer, weights) + biases


def tanh(x: ActivationType) -> ActivationType:
    """
    Applies the hyperbolic tangent activation function.

    Args:
        x: The input to the activation function.

    Returns:
        The activated output.
    """
    return np.tanh(x)  # type: ignore


def tanh_derivative(x: ActivationType) -> ActivationType:
    """
    Computes the derivative of the hyperbolic tangent activation function.

    Args:
        x: The input to the activation function.

    Returns:
        The derivative of the hyperbolic tangent function.
    """
    return 1 - np.tanh(x) ** 2  # type: ignore


def sigmoid(x: ActivationType) -> ActivationType:
    """
    Applies the sigmoid activation function.

    Args:
        x: The input to the activation function.

    Returns:
        The activated output.
    """
    return 1 / (1 + np.exp(-x))  # type: ignore


def sigmoid_derivative(x: ActivationType) -> ActivationType:
    """
    Computes the derivative of the sigmoid activation function.

    Args:
        x: The input to the activation function.

    Returns:
        The derivative of the sigmoid function.
    """
    return x * (1 - x)  # type: ignore


def relu(x: ActivationType) -> ActivationType:
    """
    Applies the ReLU activation function.

    Args:
        x: The input to the activation function.

    Returns:
        The activated output.
    """
    return np.maximum(0, x)  # type: ignore


def relu_derivative(x: ActivationType) -> ActivationType:
    """
    Computes the derivative of the ReLU activation function.

    Args:
        x: The input to the activation function.

    Returns:
        The derivative of the ReLU function.
    """
    return np.where(x > 0, 1, 0)  # type: ignore
