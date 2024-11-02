import time
import jax  # This line resolves the "jax not defined" issue
import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize
from jax.example_libraries import optimizers

def run_bfgs_optimization(action_to_minimize, p_initial):
    start_time = time.time()
    result_bfgs = minimize(action_to_minimize, p_initial, method='BFGS')
    end_time = time.time()
    action_value_bfgs = action_to_minimize(result_bfgs.x)
    return action_value_bfgs, end_time - start_time

def run_adam_optimization(action_to_minimize, p_initial, num_steps=2000, step_size=0.01):
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
    opt_state = opt_init(p_initial)
    
    @jax.jit
    def adam_step(i, opt_state):
        p = get_params(opt_state)
        value, grads = jax.value_and_grad(action_to_minimize)(p)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, value

    start_time = time.time()
    for i in range(num_steps):
        opt_state, action_value_adam = adam_step(i, opt_state)
    
    end_time = time.time()
    p_optimal_adam = get_params(opt_state)
    action_value_adam = action_to_minimize(p_optimal_adam)
    return action_value_adam, end_time - start_time

def run_newtons_method(action_to_minimize, grad_action, hessian_action, p_initial, num_steps=20):
    p_newton = p_initial
    start_time = time.time()
    for i in range(num_steps):
        grad_val = grad_action(p_newton)
        hess_val = hessian_action(p_newton)
        hess_val += jnp.eye(hess_val.shape[0]) * 1e-6  # Regularization
        delta_p = jnp.linalg.solve(hess_val, grad_val)
        p_newton = p_newton - delta_p
    end_time = time.time()
    action_value_newton = action_to_minimize(p_newton)
    return action_value_newton, end_time - start_time

def run_hessian_optimization(action_to_minimize, grad_action, hessian_action, p_initial, num_steps=20, learning_rate=0.05):
    p_hessian = p_initial
    start_time = time.time()
    for i in range(num_steps):
        grad_val = grad_action(p_hessian)
        hess_val = hessian_action(p_hessian)
        hess_val += jnp.eye(hess_val.shape[0]) * 1e-6  # Regularization
        delta_p = jnp.linalg.solve(hess_val, grad_val)
        p_hessian = p_hessian - learning_rate * delta_p
    end_time = time.time()
    action_value_hessian = action_to_minimize(p_hessian)
    return action_value_hessian, end_time - start_time

def print_final_comparison(bfgs_result, bfgs_time, adam_result, adam_time, newton_result, newton_time, hessian_result, hessian_time):
    print("\nFinal Action Values and Time Comparison:")
    print(f"BFGS: {bfgs_result} | Time Taken: {bfgs_time:.4f} seconds")
    print(f"Adam: {adam_result} | Time Taken: {adam_time:.4f} seconds")
    print(f"Newton's Method: {newton_result} | Time Taken: {newton_time:.4f} seconds")
    print(f"Hessian-based Optimization: {hessian_result} | Time Taken: {hessian_time:.4f} seconds")
