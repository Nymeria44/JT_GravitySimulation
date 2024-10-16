import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, tanh
from Schwarzian import schwarzian_action

# Var class for symbolic and numerical values
class Var:
    def __init__(self, sym_value, num_value=None):
        self.sym = sym_value  # Symbolic part
        self.num = num_value  # Numerical part

    def substitute(self, symbol, dataPoints):
        """
        Substitute a symbolic value with a numerical one.
        """
        self.num = self.sym.subs(symbol, dataPoints)

# Define symbolic variables
t = symbols('t')
a = Var(symbols('a'))
f = Var(tanh(a.sym * t))

# Integration limits
t0 = 0
t1 = 1

dataPoints = 200

# Define the range of 'a' values (numerical)
a_values = np.linspace(0.1, 5, dataPoints )

# Init list action values
action_values = []

for a_value in a_values:
    a.substitute(a.sym, a_value)
    f.substitute(a.sym, a_value)

    # Compute the Schwarzian action
    action = schwarzian_action(f.num, t, t0, t1, C=1, numerical=True)
    
    # Store the action value
    action_values.append(action)

# Plot action vs 'a'
plt.plot(a_values, action_values)
plt.xlabel('Parameter a')
plt.ylabel('Schwarzian Action')
plt.title('Schwarzian Action vs Parameter a')
plt.grid(True)
plt.show()
