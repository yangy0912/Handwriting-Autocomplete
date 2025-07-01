import numpy as np
import pandas as pd
from mathHelper import*
import matplotlib.pyplot as plt

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

def classify(pixels):
    # supress all pixels to [0, 1]
    pixels = pixels / 255.0
    # zero out neurons
    l1 = sigmoid(np.matmul(pixels, w1) + b1)
    l2 = relu(np.matmul(l1, w2) + b2)
    l3 = relu(np.matmul(l2, w3) + b3)
    l4 = relu(np.matmul(l3, w4) + b4)
    output = softmax(np.matmul(l4, op) + bO)
    return output

def classify_bundle(arr):
    # Get row data
    df = pd.read_csv('../A_Z Handwritten Data/train_df_shuffled.csv')
    classification_result = np.full_like(arr, 0)
    for index, value in enumerate(arr):
        row = df.iloc[value]
        label = row["0"]
        pixels = row.iloc[2:].to_numpy()
        op_layer = classify(pixels).ravel()
        result = find_max(op_layer)
        classification_result[index] = result
    return classification_result



def display_original(pixels):
    disp = pixels.reshape(28, 28)
    plt.imshow(disp, cmap='gray')
    plt.show()

print(classify_bundle([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]))






