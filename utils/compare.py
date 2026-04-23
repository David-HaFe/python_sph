# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd
from config import compared_files


def compare():
    width = max(len(file) for file in compared_files)
    errors = np.zeros((compared_files.size, compared_files.size))

    print(" " * (width + 3), end="")
    print(" | ".join(f"{file:<{width}}" for file in compared_files))

    for index_1, file_1 in enumerate(compared_files):
        for index_2, file_2 in enumerate(compared_files):
            df1 = pd.read_csv(f"{file_1}.csv", delimiter=",")
            df2 = pd.read_csv(f"{file_2}.csv", delimiter=",")

            data1 = np.array(
                [
                    np.fromstring(row.strip("[]").replace(",", " "), sep=" ")
                    for row in df1["data"]
                ]
            )
            data2 = np.array(
                [
                    np.fromstring(row.strip("[]").replace(",", " "), sep=" ")
                    for row in df2["data"]
                ]
            )

            t1 = df1["time"].to_numpy()
            t2 = df2["time"].to_numpy()

            # interpolate denser data series to match the less dense one
            if len(t1) >= len(t2):
                data1_interp = np.array(
                    [np.interp(t2, t1, data1[:, i]) for i in range(data1.shape[1])]
                ).T
                data2_interp = data2
            else:
                data1_interp = data1
                data2_interp = np.array(
                    [np.interp(t1, t2, data2[:, i]) for i in range(data2.shape[1])]
                ).T

            squared_error = (data1_interp - data2_interp) ** 2
            mse = np.mean(squared_error)
            errors[index_1][index_2] = mse

        print()
        print(f"{file_1:<{width}} | ", end="")
        print(" | ".join(f"{error:<{width}.5f}" for error in errors[index_1]))
