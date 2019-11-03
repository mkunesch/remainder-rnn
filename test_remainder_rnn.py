#!/usr/bin/env python3
"""Test a model that has been trained to classify integers by their remainders.

This module contains the functionality to load a model and the weights from
checkpoint files and to test the model on both a random test set and interactive
user input.
"""
import click
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from remainder_dataset import create_remainder_dataset, random_digit_array


def interactive_test(model, divisor):
    """Test a remainder classification model on user defined input.

    Tests a model that has been trained to classify integers by their remainder
    modulo a given divisor on user input. If no input is provided an integer
    is randomly generated.
    """
    test_sequence = input(
        "Enter the sequence (leave blank for randomly generated number): "
    )
    if test_sequence == "":
        print("No sequence entered. Using the random sequence:")
        number, digit_array = random_digit_array(
            max_length=1000, distribute_length=True
        )
        print(number)
    else:
        digit_array = [int(c) for c in list(test_sequence)]
        number = int(test_sequence)

    one_hot = to_categorical(digit_array, num_classes=10)
    one_hot = one_hot.reshape(1, one_hot.shape[0], one_hot.shape[1])
    print(
        "Predicted remainder: %d. Correct remainder: %d."
        % (model.predict_classes(one_hot)[0], number % divisor)
    )


def test(model, divisor, max_test_length=1000):
    """Test a model that classifies integers by their remainder.

    Args:
        model: the trained model
        divisor: the divisor with which the remainder is to be calculated
        max_test_length: the maximum number of digits of test integers (base 10)
    """
    print(f"Testing on numbers of up to {max_test_length} digits")
    test_data, test_labels = create_remainder_dataset(
        divisor, num_examples=1000, max_length=max_test_length
    )
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {test_accuracy}. Test loss: {test_loss}.")

    # Output incorrectly labelled test data
    output_labels = model.predict_classes(test_data)
    mask = np.not_equal(output_labels, np.argmax(test_labels, axis=1))
    errors = np.argmax(test_data[mask, :, :], axis=2)
    error_labels = output_labels[mask]
    correct_labels = np.argmax(test_labels[mask], axis=1)

    print("Examples of incorrect predictions:")
    for err, wrong, right in zip(errors[0:5, :], error_labels, correct_labels):
        print(f"{err} was labeled as {wrong} but should have been {right}.")


@click.command()
@click.argument("model-path", type=click.Path(dir_okay=False))
def main(model_path):
    """Load model and weights from file and test."""
    model = load_model(model_path)

    # The dimension of the last layer gives the divisor
    divisor = model.layers[-1].output_shape[-1]

    # Test the model
    test(model, divisor)

    # Interactive test. Keep going until the user gets bored and kills
    while True:
        interactive_test(model, divisor)


if __name__ == "__main__":
    main()
