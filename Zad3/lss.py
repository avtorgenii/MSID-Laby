"""
Task: implement LSS method.

Here are some useful hints:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
# https://www.statsoft.pl/textbook/stathome_stat.html?https%3A%2F%2Fwww.statsoft.pl%2Ftextbook%2Fstglm.html
"""
import random
import urllib.request
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fetch_data_file() -> str:
    """Download a file with target data to fit LSS algo in."""
    file = "GDP_happiness.csv"
    url_server = "https://byes.pl/wp-content/uploads/datasets/" + file
    if not os.path.isfile(file):
        urllib.request.urlretrieve(url_server, file)
    else:
        print(f"File already exists!")
    return file


def read_data_vectors() -> Tuple[np.ndarray, np.ndarray]:
    """Read target data and obtain X and Y vectors without NaNs."""
    gdp_happines_df = pd.read_csv(fetch_data_file(), index_col=[0])
    gdp_happines_df = gdp_happines_df.fillna(gdp_happines_df.mean(axis=0))
    gdp_happines_df.tail()

    X = gdp_happines_df["GDP per capita"].values
    Y = gdp_happines_df["happiness"].values

    return X, Y


def get_polynomial_form(polynomial_degree: int) -> np.ndarray:
    """
    Get array with form of polynomial.

    :param polynomial_degree: a degree of polynomial
        [[0], [1]] - 1st order, [[0], [1], [2]] - 2nd order, 
        [[0], [1], [2], [3]] - 3rd order, and so on...
    :return: a array with degrees of polynomial
    """

    return np.arange(polynomial_degree + 1).reshape(-1, 1)


def print_polynomial(theta: np.ndarray, precision: int = 3) -> str:
    """Return string representation of polynomial."""
    out = ""

    for i, coefficient in enumerate(theta):
        if i != len(theta) - 1:
            rounded_coefficient = round(coefficient[0], precision)
            out += f"{rounded_coefficient}*x^{i} + "
        else:
            out += f"{round(coefficient[0], precision)}*x^{i}"

    return out


def least_squares_solution(
        X: np.ndarray, Y: np.ndarray, polynomial_degree: int
) -> np.ndarray:
    """
    Compute theta matrix with coefficients of polynomial fitted by LSS.

    :param X: argument vector, shape = (N, )
    :param Y: target vector, shape = (N, ) 
    :param polynomial_degree: degree of fitted polynomial

    :return: theta matrix of polynomial, shape = (1, polynomial_degree + 1)
    """

    x_stack = np.column_stack([X ** i for i in range(polynomial_degree + 1)])

    out = np.linalg.inv(x_stack.T @ x_stack) @ x_stack.T @ Y

    print(out)

    return np.array(out).reshape(-1, 1)


def generalised_linear_model(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Compute values for generalised linear model.

    :param X: argument vector, shape = (N, )
    :param T: theta matrix of polynomial, shape = (1, polynomial_degree + 1)
    :return: regressed values, shape = (N, )
    """
    return sum([coeff * X ** degree for degree, coeff in enumerate(T)])


def visualise_LSS_method(X: np.ndarray, Y: np.ndarray, T: np.ndarray):
    """
    Visualise LSS model on fancy Matplotlib plot.

    :param X: input argument vector
    :param Y: input target vector
    :param T: theta vector with coefficients of ploynomial
    """
    X_test = np.linspace(start=X.min(), stop=X.max(), num=300)
    Y_test = np.linspace(start=Y.min(), stop=Y.max(), num=300)
    Y_pred = generalised_linear_model(X_test, T)

    es = Y_test - Y_pred
    pow_es = np.power(es, 2)
    sum_pow_es = np.sum(pow_es)
    err = np.sqrt(sum_pow_es) / len(Y)

    plt.scatter(X, Y, color="tab:blue", label="real data")
    plt.plot(X_test, Y_pred, color="tab:orange", label="estimated function")
    plt.xlabel("x - GDP", fontsize=14)
    plt.ylabel("y - happiness", fontsize=14)
    plt.title(f"Fitted: \n {print_polynomial(T, precision=5)}")
    plt.text(150, 6, f'Błąd - {round(err, 5)}', bbox=dict(facecolor='pink', alpha=0.5))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # here is a playground for your tests!
    X, Y = read_data_vectors()
    T = least_squares_solution(X, Y, 2)
    print(print_polynomial(T))
    visualise_LSS_method(X, Y, T)
