import numpy as np
import pandas as pd
from DeepLearning import *

weights = np.load('./Model/weights.npz')
w1 = weights['w1']
w2 = weights['w2']
w3 = weights['w3']
w4 = weights['w4']
op = weights['op']

def classify(pixels):
    # supress all pixels to [0, 1]
    pixels = pixels / 255.0
    # zero out neurons
    l1 = np.matmul(pixels, w1)
    l2 = np.matmul(l1, w2)
    l3 = np.matmul(l2, w3)
    l4 = np.matmul(l3, w4)
    output = np.matmul(l4, op)
    # Softmax?
    return output

df = pd.read_csv('../A_Z Handwritten Data/train_df_shuffled.csv')
row = df.iloc[0]
label = row["0"]
pixels = row.iloc[1:].to_numpy()
print("label: " + str(label))
print(len(row))



