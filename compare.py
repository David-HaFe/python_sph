# script to quickly compute MSE between two data sets
import numpy as np
import pandas as pd

files = np.array(
    [
        # "exact_heat_eq_10x10_r1_5",
        # "exact_heat_eq_15x15_r1_5",
        # "exact_heat_eq_20x20_r1_5",
        "heat_eq_10x10_r1_5",
        "heat_eq_15x15_r1_5",
        "heat_eq_20x20_r1_5",
    ]
)
width = max(len(file) for file in files)
errors = np.zeros((files.size, files.size))

print(" " * (width + 3), end="")
print(" | ".join(f"{file:<{width}}" for file in files))

for index_1, file_1 in enumerate(files):
    for index_2, file_2 in enumerate(files):
        df1 = pd.read_csv(f"csvs/{file_1}.csv", delimiter=",")
        df2 = pd.read_csv(f"csvs/{file_2}.csv", delimiter=",")

        data1 = np.array(
            [np.fromstring(
                row.strip("[]").replace(",", " "), sep=" "
            ) for row in df1["data"]]
        )
        data2 = np.array(
            [np.fromstring(
                row.strip("[]").replace(",", " "), sep=" "
            ) for row in df2["data"]]
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
