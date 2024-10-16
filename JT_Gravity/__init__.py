# __init__.py
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, tanh
from Schwarzian import schwarzian_action

# Define symbolic variables
t = symbols('t')
a_sym = symbols('a')
f_sym = tanh(a_sym * t)

# Integration limits
t0 = 0
t1 = 1

data_points = 200

# Define the range of 'a' values (numerical)
a_values = np.linspace(0.1, 5, data_points)

# Initialise list to store action values
action_values = []

for a_value in a_values:
    # Discretize the function
    f_num = f_sym.subs(a_sym, a_value)

    # Compute the Schwarzian action
    action = schwarzian_action(f_num, t, t0, t1, C=1, numerical=True)
    action_values.append(action)

# Plot action vs 'a'
plt.plot(a_values, action_values)
plt.xlabel('Parameter a')
plt.ylabel('Schwarzian Action')
plt.title('Schwarzian Action vs Parameter a')
plt.grid(True)
plt.show()
