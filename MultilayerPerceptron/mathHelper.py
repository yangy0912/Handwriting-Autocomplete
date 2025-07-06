import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def difference(result, classification):
    classification_arr = np.zeros(result.shape)
    classification_arr[0, classification - 1] = 1
    return result - classification_arr

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def find_max(arr):
    max = -np.inf
    max_index = 0
    for index, value in enumerate(arr):
        if value > max:
            max = value
            max_index = index
    return max_index

