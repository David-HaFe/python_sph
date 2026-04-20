

# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd

file_1_name = "heat_eq_10x10_r1_5"
file_2_name = "heat_eq_20x20_r1_5"
# file_2_name = "exact_solution"
df1 = pd.read_csv(f"csvs/{file_1_name}.csv", delimiter=",")
df2 = pd.read_csv(f"csvs/{file_2_name}.csv", delimiter=",")

data1 = np.array([
    np.fromstring(row.strip("[]"), sep=" ") for row in df1["data"]
])
data2 = np.array([
    np.fromstring(row.strip("[]"), sep=" ") for row in df2["data"]
])

t1 = df1["time"].to_numpy()
t2 = df2["time"].to_numpy()

# interpolate denser data series to match the less dense one
if len(t1) >= len(t2):
    data1_interp = np.array([
        np.interp(t2, t1, data1[:, i]) for i in range(data1.shape[1])
    ]).T
    data2_interp = data2
else:
    data1_interp = data1
    data2_interp = np.array([
        np.interp(t1, t2, data2[:, i]) for i in range(data2.shape[1])
    ]).T

squared_error = (data1_interp - data2_interp) ** 2
print(squared_error.shape)
mse = np.mean(squared_error)

print(f"Comparing | {file_1_name} | and | {file_2_name} |")
print(f"Mean squared error (MSE):  {mse:.6f}")


