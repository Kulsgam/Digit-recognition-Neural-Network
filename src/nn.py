import numpy as np
from typing import Tuple
from data import ModelData
from node import xavier_initialization, sigmoid, sigmoid_derivative, z, ActivationType

LAYER_DIMENSIONS = [0, 16, 16, 10]


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

    def _error(self, output: ActivationType, target: ActivationType) -> np.float64:
        return np.sum((output - target) ** 2)  # type: ignore

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

        for i in range(length):
            image = self.training_data.images[i]
            label = self.training_data.labels[i]

            self._feed_forward(image)

            expected = np.zeros(LAYER_DIMENSIONS[-1])
            expected[label] = 1.0
            error = self._error(self.layers[-1], expected)
        pass

    def test(self, testing_data: ModelData) -> float:
        return 0.0
