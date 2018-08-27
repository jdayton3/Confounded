import pandas as pd
df = pd.read_csv("mnist_matrix.csv", header=None)
df = df.T
df['Batch'] = ['A'] * 5000 + ['B'] * 5000
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df['Sample'] = df.index
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.to_csv("tidy_mnist.csv", index=False)
