#!/usr/bin/env python3

"""RNN to classify integers by their remainder given a divisor.

This module contains the functionality to create and train an RNN that
classifies arbitrarily long integers by their remainder given a divisor.
"""

import sys

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np

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
    model.add(Dense(divisor, activation='softmax'))
    if verbose:
        print(model.summary())

    return model


def compile_model(model, learning_rate=0.01, decay=0.002):
    """Compile the given model using Adam optimiser."""
    optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])


def train_model(model, divisor, callbacks=None):
    """Train a given model to classify integers by their remainder.

    Train a given RNN to classify integers by their remainder when divided by a
    given divisor. Uses several phases of training with gradually longer
    integers.
    """
    # Create validation data set
    val_data, val_labels = create_remainder_dataset(
        divisor, num_examples=1000, max_length=100)

    data, labels = create_remainder_dataset(
        divisor, num_examples=10000, max_length=10, distribute_length=True)

    epochs_per_phase = 15
    # Training phase 0 (just the short integers)
    model.fit(
        data,
        labels,
        validation_data=(val_data, val_labels),
        epochs=epochs_per_phase,
        batch_size=64,
        callbacks=callbacks)

    # Training phase >= 1 (include longer integers)
    for phase in range(1, 3):
        max_length = 10 * (phase + 1)
        start_epoch = phase * epochs_per_phase
        stop_epoch = (phase + 1) * epochs_per_phase

        # Add longer integers to the training data
        data_long, labels_long = create_remainder_dataset(
            divisor, 20000, max_length, distribute_length=False)
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
            initial_epoch=start_epoch)


class LearningRateMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        learning_rate = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        learning_rate = learning_rate / (1 + decay * keras.backend.cast(
            iterations, keras.backend.dtype(decay)))
        print("Learning rate: %f." % keras.backend.eval(learning_rate))


def main():
    """Train an RNN to classify integers by their remainder.

    The remainder is calculated with respect to a divisor that should be
    supplied to the script or to the input prompt. If a compatible file called
    checkpoint_best.hdf5 is present in the directory, it is used to initialise
    the model weights. Otherwise, training starts from scratch.
    The weights that yield the best performance on the validation set are
    written to checkpoint_best.hdf5.
    """
    if len(sys.argv) > 1:
        divisor = int(sys.argv[1])
    else:
        divisor = int(
            input("No divisor provided."
                  " Which divisor would you like to train for? "))

    #Create and save the model
    model = create_model(divisor)
    compile_model(model, learning_rate=0.01, decay=0.002)
    with open("model.json", "w") as json_file:
        json_file.write(model.to_json())

    # Set up the checkpointing functionality
    checkpoint_path = "checkpoint_best.hdf5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto')

    # If compatible checkpoint file exists use it to restart
    try:
        model.load_weights("checkpoint_best.hdf5")
    except (IOError, ValueError):
        print("No compatible checkpoint file found. Training from scratch ...")
    else:
        print("Valid checkpoint file found. Loading weights ...")

    callbacks = [checkpoint, LearningRateMonitor()]

    train_model(model, divisor, callbacks)


if __name__ == "__main__":
    main()
