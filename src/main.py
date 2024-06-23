"""
Entry point of the program.

Functions:
    main(): Handles all of the abstracted logic for the perceptron model.
"""

from data import get_training_data


def main():
    """
    Handles all of the abstracted logic for the perceptron model.
    """

    training_data = get_training_data()
    print(training_data.images.shape)
    print(training_data.labels.shape)


if __name__ == "__main__":
    main()
