import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

def gradient_descent(X, y):
    m_curr = b_curr = 0
    table_data = []
    iterations = 1000
    for i in range(iterations):
        y_predicted = m_curr * X + b_curr
        cost = (1 / len(X)) * sum([val**2 for val in (y - y_predicted)])
        md = -(2 / len(X)) * sum(X * (y - y_predicted))
        bd = -(2 / len(X)) * sum(y - y_predicted)
        m_curr = m_curr - 0.01 * md
        b_curr = b_curr - 0.01 * bd
        table_data.append([i, m_curr, b_curr, cost])

    print(
        tabulate(
            table_data,
            headers=["Iteration", "m", "b", "cost"],
            tablefmt="pretty",
            floatfmt=".4f",
            numalign="right",
        )
    )
    return table_data


X = np.array([1, 2, 3, 4, 5])

y = np.array([5, 7, 9, 11, 13])

data = gradient_descent(X, y)

plt.plot([i[0] for i in data], [i[3] for i in data])
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent")
plt.show()
