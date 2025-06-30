import pandas as pd

def initial_data_shuffle_split():
    df = pd.read_csv('../A_Z Handwritten Data/A_Z Handwritten Data.csv')
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    split_portion = int(len(df) * 0.8)
    train_df = shuffled_df.iloc[:split_portion]
    test_df = shuffled_df.iloc[split_portion:]
    train_df.to_csv('../A_Z Handwritten Data/train_df_shuffled.csv')
    test_df.to_csv('../A_Z Handwritten Data/test_df_shuffled.csv')