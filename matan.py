from scipy.optimize import minimize

"""
# Definicja funkcji celu
def objective(x):
    return (x+3) * x * (x-3)


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


"""

# Variables definition
i = 1
c1 = 0.5
x0 = 10  # Starting point
e = 0.01  # Epsilon
h = 0.03  # Step


def objective(x):
    return (x - 3)*x*(x+3)


def ci(iteration):
    return (2 ** (iteration - 1)) * c1


def gradient_of_kara_function(x, iteration):
    if 2 >= x >= -2:
        return 3 * (x ** 2) - 9
    elif x < -2:
        return 3 * (x ** 2) - 9 + ci(iteration) * (2 * x + 4)
    else:
        return 3 * (x ** 2) - 9 + ci(iteration) * (2 * x - 4)


def spadek(x_start, kara_iteration):
    limit = 100
    print(f"    Iteration 0 of spadek x = {x_start}, y = {objective(x_start)}")
    iteration = 1
    x_next = x_start - h * gradient_of_kara_function(x_start, kara_iteration)
    print(f"    Iteration {iteration} of spadek x = {x_next}, y = {objective(x_next)}")

    iteration += 1

    x_curr = x_start

    while abs(x_next - x_curr) >= e and limit >= iteration:
        x_curr = x_next
        x_next = x_next - h * gradient_of_kara_function(x_curr, kara_iteration)
        print(f"    Iteration {iteration} of spadek x = {x_next}, y = {objective(x_next)}")
        iteration += 1

    return x_next


def kara_method(x_start):
    print(f"Iteration 0 of kara x = {x_start}, y = {objective(x_start)}")
    iteration = 1
    x_next = spadek(x_start, iteration)
    print(f"Iteration {iteration} of kara x = {x_next}, y = {objective(x_next)}")

    iteration += 1

    x_curr = x_start

    while abs(x_next - x_curr) >= e:
        x_curr = x_next
        x_next = spadek(x_curr, iteration)
        print(f"Iteration {iteration} of kara x = {x_next}, y = {objective(x_next)}")
        iteration += 1

    return x_next


kara_method(x0)
