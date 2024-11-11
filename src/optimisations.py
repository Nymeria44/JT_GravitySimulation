# optimisations.py

import time
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax.example_libraries import optimizers
import optax


# Adam
NUM_STEPS_ADAM = 8000
STEP_SIZE_ADAM = 0.0001

# Optax Adam
NUM_STEPS_OPTAX_ADAM = 8000
STEP_SIZE_OPTAX_ADAM = 0.0001

# Yogi
NUM_STEPS_YOGI = 5000
STEP_SIZE_YOGI = 0.0001

# L-BFGS
NUM_STEPS_LBFGS = 1500

# AdaBelief
NUM_STEPS_ADABELIEF = 3000
STEP_SIZE_ADABELIEF = 0.0001

# Newton's Method
NUM_STEPS_NEWTON = 20

# Hessian-Based Optimization
NUM_STEPS_HESSIAN = 500
LEARNING_RATE_HESSIAN = 0.0001

####################################################################################################
# Optimization Functions
####################################################################################################
def run_bfgs_optimization(action_to_minimize, p_initial, verbose=False):
    """
    Run BFGS optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    start_time = time.time()
    if verbose:
        print("Starting BFGS optimization...")
    result_bfgs = minimize(action_to_minimize, p_initial, method='BFGS')
    end_time = time.time()
    
    action_value_bfgs = action_to_minimize(result_bfgs.x)
    return result_bfgs.x, action_value_bfgs, end_time - start_time

def run_adam_optimization(action_to_minimize, p_initial, verbose=False):
    """
    Run JAX Adam optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    opt_init, opt_update, get_params = optimizers.adam(step_size=STEP_SIZE_ADAM)
    opt_state = opt_init(p_initial)

    @jax.jit
    def adam_step(i, opt_state):
        p = get_params(opt_state)
        value, grads = jax.value_and_grad(action_to_minimize)(p)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, value

    start_time = time.time()
    for i in range(NUM_STEPS_ADAM):
        opt_state, action_value_adam = adam_step(i, opt_state)
        if verbose and i % 1000 == 0:
            print(f"Adam Step {i}, Action Value: {action_value_adam}")

    end_time = time.time()
    p_optimal_adam = get_params(opt_state)
    action_value_adam = action_to_minimize(p_optimal_adam)
    return p_optimal_adam, action_value_adam, end_time - start_time

def run_optax_adam_optimization(action_to_minimize, p_initial, verbose=False):
    """
    Run Optax Adam optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    optimizer = optax.adam(learning_rate=STEP_SIZE_OPTAX_ADAM)
    opt_state = optimizer.init(p_initial)
    
    @jax.jit
    def adam_step(opt_state, params):
        grads = jax.grad(action_to_minimize)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params
    
    start_time = time.time()
    params = p_initial
    for i in range(NUM_STEPS_OPTAX_ADAM):
        opt_state, params = adam_step(opt_state, params)
        if verbose and i % 1000 == 0:
            action_value = action_to_minimize(params)
            print(f"Optax Adam Step {i}, Action Value: {action_value}")
    
    end_time = time.time()
    action_value_adam = action_to_minimize(params)
    return params, action_value_adam, end_time - start_time

def run_newtons_method(action_to_minimize, p_initial, verbose=False):
    """
    Run Newton's method optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    p_newton = p_initial
    start_time = time.time()
    
    grad_action = jax.grad(action_to_minimize)
    hessian_action = jax.hessian(action_to_minimize)

    @jax.jit
    def newton_step(p):
        grad_val = grad_action(p)
        hess_val = hessian_action(p)
        hess_val += jnp.eye(hess_val.shape[0]) * 1e-6
        delta_p = jnp.linalg.solve(hess_val, grad_val)
        return p - delta_p

    for i in range(NUM_STEPS_NEWTON):
        p_newton = newton_step(p_newton)
        if verbose and i % 10 == 0:
            action_value = action_to_minimize(p_newton)
            print(f"Newton's Method Step {i}, Action Value: {action_value}")
    
    end_time = time.time()
    action_value_newton = action_to_minimize(p_newton)
    return p_newton, action_value_newton, end_time - start_time

def run_hessian_optimization(action_to_minimize, p_initial, verbose=False):
    """
    Run Hessian-based optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    p_hessian = p_initial
    start_time = time.time()
    
    grad_action = jax.grad(action_to_minimize)
    hessian_action = jax.hessian(action_to_minimize)

    @jax.jit
    def hessian_step(p):
        grad_val = grad_action(p)
        hess_val = hessian_action(p)
        hess_val += jnp.eye(hess_val.shape[0]) * 1e-6
        delta_p = jnp.linalg.solve(hess_val, grad_val)
        return p - LEARNING_RATE_HESSIAN * delta_p

    for i in range(NUM_STEPS_HESSIAN):
        p_hessian = hessian_step(p_hessian)
        if verbose and i % 100 == 0:
            action_value = action_to_minimize(p_hessian)
            print(f"Hessian Optimization Step {i}, Action Value: {action_value}")

    end_time = time.time()
    action_value_hessian = action_to_minimize(p_hessian)
    return p_hessian, action_value_hessian, end_time - start_time

def run_yogi_optimization(action_to_minimize, p_initial, verbose=False):
    """
    Run Yogi optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    optimizer = optax.yogi(learning_rate=STEP_SIZE_YOGI)
    opt_state = optimizer.init(p_initial)
    
    @jax.jit
    def yogi_step(opt_state, params):
        grads = jax.grad(action_to_minimize)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params
    
    start_time = time.time()
    params = p_initial
    for i in range(NUM_STEPS_YOGI):
        opt_state, params = yogi_step(opt_state, params)
        if verbose and i % 500 == 0:
            action_value = action_to_minimize(params)
            print(f"Yogi Step {i}, Action Value: {action_value}")
    
    end_time = time.time()
    action_value_yogi = action_to_minimize(params)
    return params, action_value_yogi, end_time - start_time

def run_lbfgs_optimization(action_to_minimize, p_initial, verbose=False):
    """
    Run L-BFGS optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(p_initial)
    value_and_grad_fn = optax.value_and_grad_from_state(action_to_minimize)

    @jax.jit
    def lbfgs_step(opt_state, params):
        value, grad = value_and_grad_fn(params, state=opt_state)
        updates, opt_state = optimizer.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=action_to_minimize
        )
        params = optax.apply_updates(params, updates)
        return opt_state, params, value

    start_time = time.time()
    params = p_initial

    for i in range(NUM_STEPS_LBFGS):
        opt_state, params, value = lbfgs_step(opt_state, params)
        if verbose and i % 500 == 0:
            print(f"LBFGS Step {i}, Action Value: {value}")

    end_time = time.time()
    action_value_lbfgs = action_to_minimize(params)
    return params, action_value_lbfgs, end_time - start_time

def run_adabelief_optimization(action_to_minimize, p_initial, verbose=False):
    """
    Run AdaBelief optimization.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    tuple
        (optimal_params, final_action_value, time_taken)
    """
    optimizer = optax.adabelief(learning_rate=STEP_SIZE_ADABELIEF)
    opt_state = optimizer.init(p_initial)
    
    @jax.jit
    def adabelief_step(opt_state, params):
        grads = jax.grad(action_to_minimize)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params
    
    start_time = time.time()
    params = p_initial
    for i in range(NUM_STEPS_ADABELIEF):
        opt_state, params = adabelief_step(opt_state, params)
        if verbose and i % 1000 == 0:
            action_value = action_to_minimize(params)
            print(f"AdaBelief Step {i}, Action Value: {action_value}")
    
    end_time = time.time()
    action_value_adabelief = action_to_minimize(params)
    return params, action_value_adabelief, end_time - start_time
