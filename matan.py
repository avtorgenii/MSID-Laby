import numpy as np
from scipy.optimize import minimize


# Definicja funkcji celu
def objective(x):
    return (x + 3) * x * (x - 3)


def constraint1(x):
    return -x - 2


# Definicja drugiego ograniczenia
def constraint2(x):
    return x - 2


# Początkowe wartości zmiennych
x0 = [0]

# Rozwiązanie problemu optymalizacji z ograniczeniami
solution = minimize(objective, x0, constraints=[{'type': 'ineq', 'fun': constraint1},
                                                {'type': 'ineq', 'fun': constraint2}])

# Wyświetlenie wyniku
print("Minimum funkcji celu:", solution.fun)
print("Wartość x:", solution.x)

# Variables definition
i = 1
c1 = 1
x0 = (5, 5)  # Starting point
e = 0.03  # Epsilon
h = [0.3, 0.05]  # Step


def objective(x):
    x1, x2 = x
    return 2 * x1 + 4 * x2


def ci(iteration):
    return (2 ** (iteration - 1)) * c1


def cons1(x):
    x1, x2 = x
    return 4 * x1 + 3 * x2 >= 11


def cons2(x):
    x1, x2 = x
    return 5 * x1 + 2 * x2 >= 7


def cons3(x):
    x1, x2 = x
    return x1 + 3 * x2 >= 4


def cons4(x):
    x1, x2 = x
    return x1 >= 0


def cons5(x):
    x1, x2 = x
    return x2 >= 0


def gradient_of_kara_function(x, iteration):
    res = [2, 4]

    if cons1(x) and cons2(x) and cons3(x) and cons4(x) and cons5(x):
        return np.array(res)
    if not cons1(x):
        res[0] += -4
        res[1] += -3
    if not cons2(x):
        res[0] += -5
        res[1] += -2
    if not cons3(x):
        res[0] += -1
        res[1] += -3
    if not cons4(x):
        res[0] += -1
    if not cons5(x):
        res[1] += -1

    return np.array(res) * ci(iteration)


def spadek(x_start, kara_iteration):
    limit = 100
    print(f"    Iteration 0 of spadek x = {x_start}, y = {objective(x_start)}")
    iteration = 1
    x_next = x_start - gradient_of_kara_function(x_start, kara_iteration) * h
    print(f"    Iteration {iteration} of spadek x = {x_next}, y = {objective(x_next)}")

    iteration += 1

    x_curr = x_start

    while np.all(np.abs(x_next - x_curr) >= e) and limit >= iteration:
        x_curr = x_next
        x_next = x_next - h * gradient_of_kara_function(x_curr, kara_iteration)
        print(f"    Iteration {iteration} of spadek x = {x_next}, y = {objective(x_next)}")
        iteration += 1

    return x_next


def kara_method(x_start):
    limit = 10
    print(f"Iteration 0 of kara x = {x_start}, y = {objective(x_start)}")
    iteration = 1
    x_next = spadek(x_start, iteration)
    print(f"Iteration {iteration} of kara x = {x_next}, y = {objective(x_next)}")

    iteration += 1

    x_curr = x_start

    while np.all(np.abs(x_next - x_curr) >= e) and limit >= iteration:
        x_curr = x_next
        x_next = spadek(x_curr, iteration)
        print(f"Iteration {iteration} of kara x = {x_next}, y = {objective(x_next)}")
        iteration += 1

    return x_next


kara_method(x0)

from scipy.optimize import linprog

# Objective function coefficients
c = [2, 4]

# Coefficients of inequality constraints (left-hand side)
A = [[-4, -3], [-5, -2], [-1, -3]]

# Right-hand side of inequality constraints
b = [-11, -7, -4]

# Variable bounds (non-negativity)
x_bounds = (0, None)
y_bounds = (0, None)

# Solving the problem using the simplex method
result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='simplex')

# Results
print('Optimal value of the objective function:', result.fun)
print('Optimal variable values:', result.x)
