import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def squaredDifference(result, classification):
    classification_arr = np.zeros(len(result))
    classification_arr[classification] = 1
    return np.square(np.sum(result - classification_arr))

arr = [0.1, 0.2, 0.1, 0.3, 0.4, 0.5]
print(squaredDifference(arr, 2))