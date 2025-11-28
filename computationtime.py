import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

def matmul_numpy(A, B):
    return A @ B

def matmul(A, B):
    A_row,A_column = A.shape
    b_row,b_column = B.shape
    X = np.zeros((A_row, b_column),dtype=float)
    
    if b_row != A_column:
        print("Cannot compute multiplication of matrices.")
    
    
    for i in range(A_row):
        for j in range(b_column):
            total = 0
            # iterate through A_cols since b_cols is smaller
            # would miss values for the dot product
            for k in range(A_column):
                total += A[i,k] * B[k,j]
            X[i,j] = total
    return X

A = np.array([[2.0, 1.0, 4.0], [1.0, 2.0, 2.0], [2.0, 4.0, 6.0]], dtype=float)
B = np.array([[12.0], [9.0], [22.0]], dtype=float)







# Matrix sizes to test
matrix_sizes = [2, 4, 8, 16, 32, 64, 128]

# Store results in a list
results = []

# Timing repeats
n_repeats = 10

for size in matrix_sizes:
    # Generate a random matrices of given size
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # Check you have the correct answer
    C_correct = matmul_numpy(A, B)
    C_my_code = matmul(A, B)
    assert np.allclose(C_correct, C_my_code)

    # Time the multiplication calculation using numpy
    start_time = time.perf_counter()
    for _ in range(n_repeats):
        matmul_numpy(A, B)
    end_time = time.perf_counter()

    elapsed_time_numpy = (end_time - start_time) / n_repeats

    # Time the multiplication calculation using your code
    start_time = time.perf_counter()
    for _ in range(n_repeats):
        matmul(A, B)

    end_time = time.perf_counter()

    elapsed_time_my_code = (end_time - start_time) / n_repeats

    # Store the results as a dictionary
    results.append(
        {
            "size": size,
            "time_taken_numpy": elapsed_time_numpy,
            "time_taken_my_code": elapsed_time_my_code,
        }
    )
    print(size, elapsed_time_numpy, elapsed_time_my_code)


def plotGraph(save_path = "simple_plot1.png"):

    # Extract sizes and times for plotting and analysis
    sizes = [result["size"] for result in results]
    times_numpy = [result["time_taken_numpy"] for result in results]
    times_my_code = [result["time_taken_my_code"] for result in results]

    # This automatically sets up the x,y graph according to the plotted points, sizes against time
    # plots both lines
    plt.loglog(sizes, times_numpy, marker="o", linestyle="-", label="numpy")
    plt.loglog(sizes, times_my_code, marker="o", linestyle="-", label="my code")

    # Labels and title of the x,y graph
    plt.xlabel("Matrix Size (n x n)")
    plt.ylabel("Time Taken (seconds)")
    plt.title("Time to Compute Determinant vs Matrix Size")
    # adds the plot lines to make graph easier to read
    plt.grid(True)
    # this identifies which line is numpy and my code
    plt.legend()

    # Show plot
    plt.show()

    # path to save the file and resolution of graph
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

plotGraph()
