"""
This module implements a simple neural network for image classification using
a feedforward architecture and backpropagation for training.

Classes:
    NeuralNetwork: A class representing the neural network.

Functions:
    train: Trains the neural network using the entire dataset.
    test: Tests the neural network on the provided testing data.
    _activation: Chooses an activation function to use.
    _activation_derivative: Computes the derivative of the activation function.
    _get_layer_activations: Computes the activations for a layer.
    _feed_forward: Performs a feedforward pass through the network.
    _backpropagate: Performs backpropagation to update weights and biases.
    _gradient_descent: Performs batch gradient descent, stochastic gradient descent or mini-batch gradient descent
    _batch_gd: Performs batch gradient descent
    _sgd: Performs stochastic gradient descent
    _mini_batch_gd: Performs mini-batch gradient descent
"""

from typing import Tuple
import numpy as np
from data import ModelData
from node import xavier_initialization, sigmoid, sigmoid_derivative, z, ActivationType

LAYER_DIMENSIONS = [0, 16, 16, 10]
# Used so that the model doesn't overshoot the minimum, so fewer amount of backpropogation
# passes are needed


class NeuralNetwork:
    """A neural network for training and testing on image data."""

    def __init__(
        self, training_data: ModelData, learning_rate: float = 0.01, epochs: int = 10
    ):
        """
        Initializes the neural network with training data and learning rate.

        Args:
            training_data: The training data containing images and labels.
            learning_rate: The learning rate for the gradient descent algorithm.
        """
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.epochs = epochs
        LAYER_DIMENSIONS[0] = (
            training_data.images.shape[1] * training_data.images.shape[2]
        )  # Image dimensions flattened
        self.layers: list[np.ndarray[Tuple[int], np.dtype[np.float64]]] = [
            np.zeros(dimension) for dimension in LAYER_DIMENSIONS
        ]
        self.all_biases: list[np.ndarray[Tuple[int], np.dtype[np.float64]]] = [
            np.zeros(dimension) for dimension in LAYER_DIMENSIONS[1:]
        ]  # Every neuron except input layer
        self.all_weights = [
            xavier_initialization((LAYER_DIMENSIONS[i], LAYER_DIMENSIONS[i + 1]))
            for i in range(len(LAYER_DIMENSIONS) - 1)
        ]

    def train(self):
        """
        Trains the neural network using the entire dataset.
        """
        for i in range(self.epochs):
            print("\nEpoch: ", i, "\n")
            self._gradient_descent()

    def test(self, testing_data: ModelData) -> float:
        """
        Tests the neural network on the provided testing data.

        Args:
            testing_data: The testing data containing images and labels.

        Returns:
            The accuracy of the neural network on the testing data.
        """

        num_correct = 0
        length = testing_data.images.shape[0]

        for i in range(length):
            image = testing_data.images[i]
            label = testing_data.labels[i]

            self._feed_forward(image)

            if np.argmax(self.layers[-1]) == label:
                num_correct += 1

        return num_correct / length

    def _activation(self, x: ActivationType):
        return sigmoid(x)

    def _activation_derivative(self, x: ActivationType):
        return sigmoid_derivative(x)

    def _get_layer_activations(
        self,
        prev_layer: np.ndarray[Tuple[int], np.dtype[np.float64]],
        weights: np.ndarray[Tuple[int, int], np.dtype[np.float64]],
        biases: np.ndarray[Tuple[int], np.dtype[np.float64]],
    ):
        return self._activation(z(prev_layer, weights, biases))

    def _feed_forward(self, image: np.ndarray[Tuple[int, int], np.dtype[np.float64]]):
        self.layers[0] = self._activation(image.flatten())

        for i in range(1, len(LAYER_DIMENSIONS)):
            self.layers[i] = self._get_layer_activations(
                self.layers[i - 1], self.all_weights[i - 1], self.all_biases[i - 1]
            )

    def _backpropogate(
        self,
        d_error_activation: np.ndarray[Tuple[int], np.dtype[np.float64]],
        curr_layer_idx: int,
    ):
        if curr_layer_idx == 0:
            return

        d_z_weights = np.atleast_2d(self.layers[curr_layer_idx - 1])  # Shape: (1, y)
        d_z_biases = 1
        d_activation_z = self._activation_derivative(
            self.layers[curr_layer_idx]
        )  # Shape: (x,)

        delta = d_activation_z * d_error_activation  # Shape: (x,)
        delta_2D = np.atleast_2d(delta)  # Shape: (1, x)

        # print(
        #     self.all_weights[curr_layer_idx - 1].shape,
        #     d_z_weights.shape,
        #     d_activation_z.shape,
        #     delta.shape,
        #     delta_2D.shape,
        # )

        d_error_weights = np.dot(d_z_weights.T, delta_2D)  # Shape: (y, x)
        d_error_biases = d_z_biases * delta  # Shape: (x,)

        # Calculating the gradient of the error of the previous layer (d_error/d_activation)
        # This is so that it can be used to calculate d_error/d_weight

        d_prev_error_activation = np.dot(
            self.all_weights[curr_layer_idx - 1], delta
        )  # Shape: (y,)
        # print(d_prev_error_activation.shape)
        # exit()
        # Now use this to recursively calculate the gradient of the error of the previous layer
        # What I mean is to repeat

        # The gradient gives the steepest increase so you negate it to reach the minimum
        # Update weights and biases
        self.all_weights[curr_layer_idx - 1] -= self.learning_rate * d_error_weights
        self.all_biases[curr_layer_idx - 1] -= self.learning_rate * d_error_biases

        self._backpropogate(d_prev_error_activation, curr_layer_idx - 1)

    def _batch_gd(self):
        # Initially I train using the entire dataset instead of SGD or mini-batch
        length = self.training_data.images.shape[0]

        for i in range(length):
            image = self.training_data.images[i]
            label = self.training_data.labels[i]

            self._feed_forward(image)

            expected = np.zeros(LAYER_DIMENSIONS[-1])
            expected[label] = 1.0

            # d_error/d_weight = d_z/d_weight * d_activation/d_z * d_error/d_activation
            # d_error/d_bias = d_z/d_bias * d_activation/d_z * d_error/d_activation

            # d_z/d_weight = a_prev
            # d_z/d_bias = 1
            # d_activation/d_z = sigmoid_derivative(z)
            # d_error/d_activation = 2 * (output - target)

            # Imagine the (curr_layer - 1) layer has y nodes and the curr_layer layer has x nodes

            # Iterate over all of the weights and biases
            # Calculate the gradient for each weight and bias using the error
            # Update the weight and bias using the gradient

            # You can use numpy instead of iterating it all over (just simply think of it as one)
            # For example the derivative of z_l with respect to w_l is a_(l-1)
            curr_layer_idx = len(LAYER_DIMENSIONS) - 1
            d_error_activation = 2 * (
                self.layers[curr_layer_idx] - expected
            )  # Shape: (x,)

            self._backpropogate(d_error_activation, curr_layer_idx)

    def _mini_batch_gd(self):
        pass

    def _sgd(self):
        pass

    def _gradient_descent(self):
        return self._batch_gd()
