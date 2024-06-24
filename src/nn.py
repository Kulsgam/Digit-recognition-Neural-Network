import numpy as np
from data import ModelData
from node import xavier_initialization, sigmoid, sigmoid_derivative, ActivationType

LAYER_DIMENSIONS = [0, 16, 16, 10]


class NeuralNetwork:
    def __init__(self, training_data: ModelData):
        self.training_data = training_data
        LAYER_DIMENSIONS[0] = (
            training_data.images.shape[1] * training_data.images.shape[2]
        )  # Image dimensions flattened
        self.layers = [np.zeros(dimension) for dimension in LAYER_DIMENSIONS]
        self.biases = [
            np.zeros(dimension) for dimension in LAYER_DIMENSIONS[1:]
        ]  # Every neuron except input layer
        self.weights = [
            xavier_initialization((LAYER_DIMENSIONS[i], LAYER_DIMENSIONS[i + 1]))
            for i in range(len(LAYER_DIMENSIONS) - 1)
        ]

    def _activation(self, x: ActivationType):
        return sigmoid(x)

    def _activation_derivative(self, x: ActivationType):
        return sigmoid_derivative(x)

    def test(self, testing_data: ModelData) -> float:
        return 0.0

    def train(self):
        pass
