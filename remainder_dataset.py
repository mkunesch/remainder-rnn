"""Functionality for creating labelled data for classification by remainder.

This module contains the functionality for creating training data consisting of
integers labelled by their remainder modulo a given divisor.
"""

import math
import random
import numpy as np
from keras.preprocessing import sequence


def random_digit_array(max_length, distribute_length=False):
    """Create a random integer and an array of its digits base 10.

    Args:
        max_length: the maximum number of digits any integer should have
        distribute_length: if True the length of the integers has uniform
            random distribution.

    Returns:
        The random number and an array of its digits
    """
    if distribute_length:
        length = random.randint(1, max_length)
    else:
        length = max_length

    number = random.randint(1, 10**length)
    digit_array = [(number // 10**i) % 10
                   for i in range(math.floor(math.log10(number)), -1, -1)]
    return number, digit_array


def create_remainder_dataset(divisor,
                             num_examples,
                             max_length,
                             distribute_length=False):
    """Create a dataset of integers labelled by their remainder for a given divisor.

    Args:
        divisor: the divisor which should be used to calculate the remainder
        num_examples: the desired number of examples in the dataset
        max_length: the maximum number of digits any integer should have
        distribute_length: if True the length of the integers has uniform random
            distribution.

    Returns:
        Numpy arrays of examples and labels
    """
    examples = np.zeros((num_examples, max_length, 10))
    labels = np.zeros((num_examples, divisor))

    for i in range(num_examples):
        number, digit_array = random_digit_array(max_length, distribute_length)
        digit_array = sequence.pad_sequences(
            [digit_array], maxlen=max_length)[0, :]

        remainder = number % divisor
        labels[i, remainder] = 1

        for j in range(max_length):
            examples[i, j, digit_array[j]] = 1

    return examples, labels
