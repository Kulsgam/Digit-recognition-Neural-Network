import numpy as np
from typing import Tuple
from data import ModelData
from node import xavier_initialization, sigmoid, sigmoid_derivative, z, ActivationType

LAYER_DIMENSIONS = [0, 16, 16, 10]
LEARNING_RATE = 0.01  # Used so that the model doesn't overshoot the minimum, so fewer amount of backpropogation passes are needed


class NeuralNetwork:
    def __init__(self, training_data: ModelData):
        self.training_data = training_data
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

    def train(self):
        # Initially I train using the entire dataset instead of SGD or mini-batch
        length = self.training_data.images.shape[0]

        # TODO: Recursively call this and multiple backpropogation iterations

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

            # Imagine the -2 layer has y nodes and the -1 layer has x nodes

            d_z_weights = np.atleast_2d(self.layers[-2])  # Shape: (y, 1)
            d_z_biases = 1
            d_activation_z = self._activation_derivative(self.layers[-1])  # Shape: (x,)
            d_error_activation = 2 * (self.layers[-1] - expected)  # Shape: (x,)

            delta = d_activation_z * d_error_activation  # Shape: (x,)
            delta_2D = np.atleast_2d(delta)  # Shape: (x, 1)

            d_error_weights = np.dot(d_z_weights, delta_2D.T)  # Shape: (y, x)
            d_error_biases = d_z_biases * delta  # Shape: (x,)

            # Calculating the gradient of the error of the previous layer (d_error/d_activation)
            # This is so that it can be used to calculate d_error/d_weight

            d_prev_error_activation = np.dot(self.all_weights[-1], delta)  # Shape: (y,)
            # Now use this to recursively calculate the gradient of the error of the previous layer
            # What I mean is to repeat

            # The gradient gives the steepest increase so you negate it to reach the minimum
            # Update weights and biases
            self.all_weights[-1] -= LEARNING_RATE * d_error_weights
            self.all_biases[-1] -= LEARNING_RATE * d_error_biases

            # Iterate over all of the weights and biases
            # Calculate the gradient for each weight and bias using the error
            # Update the weight and bias using the gradient

            # You can use numpy instead of iterating it all over (just simply think of it as one)
            # For example the derivative of z_l with respect to w_l is a_(l-1)
        pass

    def _backpropogate(
        self,
        d_error_activation: np.ndarray[Tuple[int], np.dtype[np.float64]],
        curr_layer_idx: int,
    ):
        pass

    def test(self, testing_data: ModelData) -> float:
        return 0.0
