

# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd

df1 = pd.read_csv("csvs/heat_eq_20x20_r1_5.csv", delimiter=",")
df2 = pd.read_csv("csvs/heat_eq_10x10_r3_0.csv", delimiter=",")

data1 = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in df1["data"]])
data2 = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in df2["data"]])

squared_error = (data1 - data2) ** 2
mse = np.mean(squared_error)

print(f"Mean squared error (MSE):  {mse:.6f}")


