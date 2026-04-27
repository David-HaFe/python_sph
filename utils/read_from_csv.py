

# this file breaks down a csv back into its components

import numpy as np
import pandas as pd

# read in csv file and receive all components
def get_values_from_csv(filepath):
    data_frame = pd.read_csv(f"{filepath}.csv", delimiter=",")
    # def parse_array(s):
    #     return np.fromstring(s.strip().strip("[]"), sep=" ")

    parse = lambda entry: np.array([np.fromstring(s.strip().strip("[]"), sep=" ") for s in entry])
    t = data_frame["time"].to_numpy()
    x = parse(data_frame["x"])
    y = parse(data_frame["y"])
    data_1 = parse(data_frame["data"])
    data_2 = parse(data_frame["data"])

    # check if dimensions match up
    assert t.shape[0] == x.shape[0]
    assert t.shape[0] == y.shape[0]
    assert t.shape[0] == data_1.shape[0]
    assert t.shape[0] == data_2.shape[0]

    assert x.shape == y.shape

    return t, x, y, data_1, data_2

# def interpolate_time():
#
def interpolate_locations():
    if len(x1[0]) >= len(x2[0]):
        # interpolate data1 onto the x,y positions of data2
        data1_interp = np.array(
            [griddata((x1[i], y1[i]), data1[i], (x2[i], y2[i]), method="linear") for i in range(len(t1))]
        )
        data2_interp = data2
    else:
        # interpolate data2 onto the x,y positions of data1
        data1_interp = data1
        data2_interp = np.array(
            [griddata((x2[i], y2[i]), data2[i], (x1[i], y1[i]), method="linear") for i in range(len(t2))]
        )

# df1 = pd.read_csv(f"{file_1}.csv", delimiter=",")
# df2 = pd.read_csv(f"{file_2}.csv", delimiter=",")
#
# data1 = np.array(
#     [
#         np.fromstring(row.strip("[]").replace(",", " "), sep=" ")
#         for row in df1["data"]
#     ]
# )
# data2 = np.array(
#     [
#         np.fromstring(row.strip("[]").replace(",", " "), sep=" ")
#         for row in df2["data"]
#     ]
# )
#
# if len(x1[0]) >= len(x2[0]):
#     # interpolate data1 onto the x,y positions of data2
#     data1_interp = np.array(
#         [griddata((x1[i], y1[i]), data1[i], (x2[i], y2[i]), method="linear") for i in range(len(t1))]
#     )
#     data2_interp = data2
# else:
#     # interpolate data2 onto the x,y positions of data1
#     data1_interp = data1
#     data2_interp = np.array(
#         [griddata((x2[i], y2[i]), data2[i], (x1[i], y1[i]), method="linear") for i in range(len(t2))]
#     )
#
# t1 = df1["time"].to_numpy()
# t2 = df2["time"].to_numpy()
#
# # interpolate denser data series to match the less dense one
# if len(t1) >= len(t2):
#     data1_interp = np.array(
#         [np.interp(t2, t1, data1[:, i]) for i in range(data1.shape[1])]
#     ).T
#     data2_interp = data2
# else:
#     data1_interp = data1
#     data2_interp = np.array(
#         [np.interp(t1, t2, data2[:, i]) for i in range(data2.shape[1])]
#     ).T
#
