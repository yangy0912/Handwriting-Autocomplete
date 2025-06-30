import numpy as np

# Constants
input_size: int = 784
output_size: int = 26

# Experimental knobs
hidden_layer_nodes: int = int((input_size + output_size) / 2)
hidden_layer_size: list[int] = [64, 64, 32, 32]
learning_rate: float = 0.01
bias: list[int] = [64, 64, 32, 32]

# Neuron Container
first_layer_neurons= np.zeros(hidden_layer_size[0])
second_layer_neurons = np.zeros(hidden_layer_size[1])
third_layer_neurons = np.zeros(hidden_layer_size[2])
fourth_layer_neurons = np.zeros(hidden_layer_size[3])

def initialize_weights():
    scale = 1
    initial_weight1 = np.random.rand(hidden_layer_size[0]) * scale
    initial_weight2 = np.random.rand(hidden_layer_size[1]) * scale
    initial_weight3 = np.random.rand(hidden_layer_size[2]) * scale
    initial_weight4 = np.random.rand(hidden_layer_size[3]) * scale
    np.savez('./Model/weights.npz', w1=initial_weight1,
             w2=initial_weight2, w3=initial_weight3, w4=initial_weight4)

weights = np.load('./Model/weights.npz')
print(weights['w1'])