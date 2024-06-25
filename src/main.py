"""
Entry point of the program.

Functions:
    main(): Handles all of the abstracted logic for the perceptron model.
"""

from data import get_training_data, get_testing_data
from nn import NeuralNetwork


def main():
    """
    Handles all of the abstracted logic for the perceptron model.
    """

    nn = NeuralNetwork(get_training_data())
    nn.train()
    print("Accuracy: ", nn.test(get_testing_data()))


if __name__ == "__main__":
    main()
