import numpy as np
import pandas as pd
from mathHelper import *
from numpy.f2py.auxfuncs import throw_error

# Constants
input_size: int = 784
output_size: int = 26

# Experimental knobs
hidden_layer_nodes: int = int((input_size + output_size) / 2)
hidden_layer_size: list[int] = [64, 64, 32, 32]
learning_rate: float = 0.01
bias: list[int] = [64, 64, 32, 32]

# Neuron Container
first_layer_neurons= np.zeros((hidden_layer_size[0], 1))
second_layer_neurons = np.zeros((hidden_layer_size[1], 1))
third_layer_neurons = np.zeros((hidden_layer_size[2], 1))
fourth_layer_neurons = np.zeros((hidden_layer_size[3], 1))
output_neurons = np.zeros((output_size, 1))

def initialize_weights():
    print("Initializing weights")
    scale = 1
    initial_weight1 = np.random.uniform(low=-1, high=1, size=(784, hidden_layer_size[0])) * scale
    initial_weight2 = np.random.uniform(low=-1, high=1, size=(64, hidden_layer_size[1])) * scale
    initial_weight3 = np.random.uniform(low=-1, high=1, size=(64, hidden_layer_size[2])) * scale
    initial_weight4 = np.random.uniform(low=-1, high=1, size=(32, hidden_layer_size[3])) * scale
    initial_output_weight = np.random.uniform(low=-1, high=1, size=(32, 26)) * scale
    np.savez('./Model/weights.npz', w1=initial_weight1,
             w2=initial_weight2, w3=initial_weight3,
             w4=initial_weight4, op=initial_output_weight)

def initialize_biases():
    print("Initializing biases")
    scale = 1
    initial_bias1 = np.zeros((1, bias[0])) * scale
    initial_bias2 = np.zeros((1, bias[1])) * scale
    initial_bias3 = np.zeros((1, bias[2])) * scale
    initial_bias4 = np.zeros((1, bias[3])) * scale
    initial_output_bias = np.zeros((1, output_size)) * scale
    np.savez('./Model/biases.npz', b1=initial_bias1,
             b2=initial_bias2, b3=initial_bias3, b4=initial_bias4, bO=initial_output_bias)


def batch_training(low, high):
    if low >= high:
        throw_error('low must be smaller than high')

    print("Loading weights and biases...")
    # Initialize from Data
    training_df = pd.read_csv('../A_Z Handwritten Data/train_df_shuffled.csv')
    weights = np.load('./Model/weights.npz')
    biases = np.load('./Model/biases.npz')

    b1 = biases['b1']
    b2 = biases['b2']
    b3 = biases['b3']
    b4 = biases['b4']
    bO = biases['bO']

    w1 = weights['w1']
    w2 = weights['w2']
    w3 = weights['w3']
    w4 = weights['w4']
    op = weights['op']

    percentage = set()
    print("Training...")
    # Iterate through range in batches of 10
    for batch in range(low, high + 1, 10):
        for index in range(0, 10):
            i = batch + index
            progress = (i - low) / (high - low)
            progress_percent = np.round(progress * 100)
            if progress_percent not in percentage:
                percentage.add(progress_percent)
                print("Progress: " + str(progress_percent) + "%")
            row = training_df.iloc[i]
            label = row["0"]
            pixels = row.iloc[2:].to_numpy()
            squished_pixels = pixels / 255.0

            # Forward pass
            z1 = np.matmul(pixels, w1) + b1
            a1 = sigmoid(z1)

            z2 = np.matmul(a1, w2) + b2
            a2 = relu(z2)

            z3 = np.matmul(a2, w3) + b3
            a3 = relu(z3)

            z4 = np.matmul(a3, w4) + b4
            a4 = relu(z4)

            z5 = np.matmul(a4, op) + bO
            output = softmax(z5)

            # Back Propagation
            dz5 = difference(output, label)
            d_op = np.matmul(a4.T, dz5)

            d_bO = np.sum(dz5, axis=0)

            # Layer 4
            da4 = np.matmul(dz5, op.T)
            dz4 = da4 * relu_deriv(z4)
            d_w4 = np.matmul(a3.T, dz4)
            d_b4 = np.sum(dz4, axis=0)

            # Layer 3
            da3 = np.matmul(dz4, w4.T)
            dz3 = da3 * relu_deriv(z3)
            d_w3 = np.matmul(a2.T, dz3)
            d_b3 = np.sum(dz3, axis=0)

            # Layer 2
            da2 = np.matmul(dz3, w3.T)
            dz2 = da2 * relu_deriv(z2)
            d_w2 = np.matmul(a1.T, dz2)
            d_b2 = np.sum(dz2, axis=0)

            # Layer 1
            da1 = np.matmul(dz2, w2.T)
            dz1 = da1 * sigmoid_deriv(z1)
            d_w1 = np.matmul(pixels.reshape((784, 1)), dz1)
            d_b1 = np.sum(dz1, axis=0)

            # Output layer
            op -= learning_rate * d_op
            bO -= learning_rate * d_bO

            # Layer 4
            w4 -= learning_rate * d_w4
            b4 -= learning_rate * d_b4

            # Layer 3
            w3 -= learning_rate * d_w3
            b3 -= learning_rate * d_b3

            # Layer 2
            w2 -= learning_rate * d_w2
            b2 -= learning_rate * d_b2

            # Layer 1
            w1 -= learning_rate * d_w1
            b1 -= learning_rate * d_b1
    print("Finished training")
    # Save weights and biases
    np.savez('./Model/weights.npz', w1=w1, w2=w2, w3=w3, w4=w4, op=op)
    np.savez('./Model/biases.npz', b1=b1, b2=b2, b3=b3, b4=b4, bO=bO)
    print("Saved weights and biases")

# Train
def train():
    initialize_weights()
    initialize_biases()
    batch_training(0, 290000)

initialize_weights()
initialize_biases()

