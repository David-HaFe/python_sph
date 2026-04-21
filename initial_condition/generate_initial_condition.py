from config import x_limit, y_limit
import csv
import numpy as np

path = "initial_condition/initial_condition.csv"
with open(path, "w+", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    row = np.zeros(x_limit)

    for i in range(0, y_limit):
        writer.writerow(row)
