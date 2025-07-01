import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    if x <= 0:
        return 0
    else:
        return 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def squared_difference(result, classification):
    classification_arr = np.zeros(len(result))
    classification_arr[classification] = 1
    return np.square(np.sum(result - classification_arr))

def softmax(x):
    e_x = np.exp(x - np.max(x))  # stabilize
    return e_x / e_x.sum()

def find_max(arr):
    max = -np.inf
    max_index = 0
    for index, value in enumerate(arr):
        if value > max:
            max = value
            max_index = index
    return max_index

