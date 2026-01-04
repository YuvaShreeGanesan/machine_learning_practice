import numpy as np

def gradient_desc(x, y):
    m_curr = 0
    b_curr = 0
    iterations = 2000
    n = len(x)
    learning_rate = 0.0001   # safe learning rate

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr

        cost = (1/n) * np.sum((y - y_predicted) ** 2)

        md = -(2/n) * np.sum(x * (y - y_predicted))
        bd = -(2/n) * np.sum(y - y_predicted)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        if i % 200 == 0:
            print(f"iteration={i}, m={m_curr:.4f}, b={b_curr:.4f}, cost={cost:.2f}")

# Data
math = np.array([92,56,88,70,80,49,65,35,66,67])
cs   = np.array([98,68,81,80,83,52,66,30,68,73])

# Call function
gradient_desc(math, cs)
