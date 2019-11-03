#!/usr/bin/env python3

"""RNN to classify integers by their remainder given a divisor.

This module contains the functionality to create and train an RNN that
classifies arbitrarily long integers by their remainder given a divisor.
"""

import click
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence

from remainder_dataset import create_remainder_dataset


def create_model(divisor, verbose=True):
    """Create an RNN that can learn to classify integers by their remainder.

    Args:
        divisor: the divisor with respect to which the model should learn to
            calculate remainders.
        verbose: if True, the model summary is printed.
    """
    model = Sequential()
    model.add(TimeDistributed(Dense(32), input_shape=(None, 10)))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(Dense(32)))
    model.add(Dropout(0.1))
    model.add(LSTM(5 * divisor))
    model.add(Dense(divisor, activation="softmax"))
    if verbose:
        model.summary()

    return model


def compile_model(model, learning_rate=0.01, decay=0.002):
    """Compile the given model using Adam optimiser."""
    optimizer = Adam(lr=learning_rate, decay=decay)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )


def train_model(model, divisor, callbacks=None):
    """Train a given model to classify integers by their remainder.

    Train a given RNN to classify integers by their remainder when divided by a
    given divisor. Uses several phases of training with gradually longer
    integers.
    """
    # Create validation data set
    val_data, val_labels = create_remainder_dataset(
        divisor, num_examples=1000, max_length=100
    )

    data, labels = create_remainder_dataset(
        divisor, num_examples=10000, max_length=10, distribute_length=True
    )

    epochs_per_phase = 15
    # Training phase 0 (just the short integers)
    model.fit(
        data,
        labels,
        validation_data=(val_data, val_labels),
        epochs=epochs_per_phase,
        batch_size=64,
        callbacks=callbacks,
    )

    # Training phase >= 1 (include longer integers)
    for phase in range(1, 3):
        max_length = 10 * (phase + 1)
        start_epoch = phase * epochs_per_phase
        stop_epoch = (phase + 1) * epochs_per_phase

        # Add longer integers to the training data
        data_long, labels_long = create_remainder_dataset(
            divisor, 20000, max_length, distribute_length=False
        )
        data = sequence.pad_sequences(data, max_length)
        data = np.concatenate([data, data_long], axis=0)
        labels = np.concatenate([labels, labels_long], axis=0)

        model.fit(
            data,
            labels,
            validation_data=(val_data, val_labels),
            epochs=stop_epoch,
            batch_size=64,
            callbacks=callbacks,
            initial_epoch=start_epoch,
        )


class LearningRateMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        learning_rate = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        iterations = keras.backend.cast(iterations, keras.backend.dtype(decay))
        learning_rate = learning_rate / (1 + decay * iterations)
        print(f"Learning rate: {learning_rate}.")


@click.command()
@click.argument("model-path", type=click.Path(dir_okay=False))
@click.option(
    "-d",
    "--divisor",
    type=int,
    required=True,
    help="Divisor to train the model for.",
)
def main(model_path, divisor):
    """Train an RNN to classify integers by their remainder.

    The remainder is calculated with respect to the provided divisor. The
    provided model path is used to store the model whose performance on the
    validation set is best. If the file already exists it is loaded and
    training continues from the loaded weights.
    """
    model = create_model(divisor)
    compile_model(model, learning_rate=0.01, decay=0.002)

    # Set up the checkpointing functionality
    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
    )

    # If a compatible checkpoint file was provided use it to restart
    try:
        model.load_weights(model_path)
    except (IOError, ValueError):
        print("No compatible weights provided. Training from scratch.")

    callbacks = [checkpoint, LearningRateMonitor()]

    train_model(model, divisor, callbacks)


if __name__ == "__main__":
    main()
