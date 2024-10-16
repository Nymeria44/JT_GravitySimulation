import torch
import torchquad
from torchquad import Trapezoid, set_up_backend

set_up_backend("torch", "cuda")

# Define the function f(u)
def f(u):
    return torch.sin(u) + 0.1 * u

# Define the Schwarzian derivative using PyTorch's automatic differentiation
def schwarzian_action(u):
    # Enable PyTorch autograd
    u = u.requires_grad_(True)
    
    # Compute f'(u), f''(u), and f'''(u)
    f_u = f(u)
    f_prime = torch.autograd.grad(f_u, u, create_graph=True)[0]
    f_double_prime = torch.autograd.grad(f_prime, u, create_graph=True)[0]
    f_triple_prime = torch.autograd.grad(f_double_prime, u, create_graph=True)[0]
    
    # Calculate the Schwarzian derivative
    schwarzian = (f_triple_prime / f_prime) - (3/2) * (f_double_prime / f_prime)**2
    
    return schwarzian

# Define the range of integration
u_start = 0
u_end = torch.pi  # Example range, can be adjusted

# Define the quadrature method (Trapezoid is simple and robust)
integrator = Trapezoid()

# Perform the integration
schwarzian_integral = integrator.integrate(schwarzian_action, dim=1, integration_domain=[[u_start, u_end]])

# Output the result
print(f"Schwarzian Action: {schwarzian_integral.item()}")
