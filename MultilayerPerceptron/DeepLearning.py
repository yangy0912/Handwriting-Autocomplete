import numpy as np
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

    for batch in range(low, high + 1, 10):
        for index in range(0, 10):
            print(str(batch + index))
        print("next row")

batch_training(11, 120)

'''
initialize_weights()
weights = np.load('./Model/weights.npz')
print(np.average(weights['w1']))
print(np.average(weights['w2']))
print(np.average(weights['w3']))
print(np.average(weights['w4']))
print(np.average(weights['op']))

initialize_biases()
biases = np.load('./Model/biases.npz')
print(np.average(biases['b1']))
print(np.average(biases['b2']))
print(np.average(biases['b3']))
print(np.average(biases['b4']))
print(np.average(biases['bO']))
'''
