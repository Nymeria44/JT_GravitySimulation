from sympy import symbols, tanh
from Schwarzian import schwarzian_action

# Define the symbolic variable
x = symbols('x')

# Define the function f(x)
f = tanh(x)

# Define the integration limits
t0 = 0
t1 = 1

# Compute the Schwarzian action
action = schwarzian_action(f, x, t0, t1, C=1, numerical=True)

# Display the result
print("The Schwarzian action of tanh(x) from t=0 to t=1 is:")
print(action)
