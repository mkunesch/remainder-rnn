# Remainder RNN

A small practice project with RNNs: classifying arbitrarily long integers by their remainder
modulo a given divisor.

Disclaimer: I am not an expert on machine learning and only performed this experiment
to get an intuition for some of the content in Andrew Ng's Deep Learning course
on Coursera.

Feedback, suggestions, and criticism are very welcome.

## Mathematical background
When trying to calculate the remainder of an integer modulo a given divisor,
there are several tricks, which are all based on modular arithmetic.
For example,
for 11 the procedure is:

![Modulo 11 calculation](modulo_11_calculation.png "Modulo 11"),

where `d_i` are the digits of the number base 10. Thus, finding the remainder
modulo 11 is equivalent to taking the sum of digits with alternating signs.
Exactly the same procedure can be followed for a more general divisor `p`: `10^i` will always
be a sequence of numbers with period of at most `p`, so that the remainder of a
number modulo p can be found by taking an appropriately weighted sum of the
digits. In theory, even a simple RNN should be able to learn this trick.

## Setup
In this experiment, we try to get a simple RNN to classify arbitrarily long
integers by their remainder modulo a given divisor, `p`. As input, we use
arbitrarily long integers, represented as sequence of one-hot encoded digits.
The RNN consists of 2 time-distributed
dense layers, one LSTM layer with `5*p` units, and a dense output layer with
softmax activation functions and output dimension `p`.
For complicated divisors such as 13, it is necessary to add dropout with rate
0.1 to the first two dense layers to prevent overfitting.
To train the model, we gradually increase the length of the integers, starting
with 10 digits and finishing with 30.

## Results
By the mathematical intuition above, the complexity of the task depends on the
period of the sequence `10^i` modulo `p`.
The most complicated divisor I managed to train the RNN on was 13.
When the training converges, the model always
has near-perfect accuracy and generalises well. For example,
for 13, after training for 45 epochs on 50,000 30-digit numbers, it correctly
classified all 1000-digit integers in a test set of 10,000 examples.

## Current experiments
Since for more complicated divisors, the training process only converges if the
length of integers is increased gradually, it would be very interesting to try
Teacher-Student Curriculum Learning (Matiisen et al.,
https://arxiv.org/abs/1707.00183). This is work in progress.

## Usage
The scripts require Keras to run. Run e.g. `./remainder_rnn.py 13` to train for the
divisor 13, then `./test_remainder_rnn.py` to test.
