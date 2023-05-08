import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    l = df['label'].values.tolist()
    i = df['seq'].values.tolist()
    labels = []
    inputs = []
    # print(l)
    # print(i)
    # breakpoint()
    for _ in range(10000):
        labels.extend(l)
        inputs.extend(i)

    new_df = pd.DataFrame()
    new_df['seq'] = inputs
    new_df['label'] = labels
    new_df.to_csv('data.train.csv', index=False)