import time
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax.example_libraries import optimizers
import optax

# Fine-Tuning Parameters
NUM_STEPS_ADAM = 3000
STEP_SIZE_ADAM = 0.001

NUM_STEPS_OPTAX_ADAM = 3000
STEP_SIZE_OPTAX_ADAM = 0.003

NUM_STEPS_YOGI = 3000
STEP_SIZE_YOGI = 0.002

NUM_STEPS_LBFGS = 3000
STEP_SIZE_LBFGS = 0.01

NUM_STEPS_ADABELIEF = 3000
STEP_SIZE_ADABELIEF = 0.001

NUM_STEPS_NEWTON = 10

NUM_STEPS_HESSIAN = 40
LEARNING_RATE_HESSIAN = 0.001

# Optimization Functions
def run_bfgs_optimization(action_to_minimize, p_initial):
    start_time = time.time()
    result_bfgs = minimize(action_to_minimize, p_initial, method='BFGS')
    end_time = time.time()
    action_value_bfgs = action_to_minimize(result_bfgs.x)
    return result_bfgs.x, action_value_bfgs, end_time - start_time

def run_adam_optimization(action_to_minimize, p_initial):
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
    
    end_time = time.time()
    p_optimal_adam = get_params(opt_state)
    action_value_adam = action_to_minimize(p_optimal_adam)
    return p_optimal_adam, action_value_adam, end_time - start_time

def run_optax_adam_optimization(action_to_minimize, p_initial):
    optimizer = optax.adam(learning_rate=STEP_SIZE_OPTAX_ADAM)
    opt_state = optimizer.init(p_initial)
    
    @jax.jit
    def adam_step(i, opt_state, params):
        grads = jax.grad(action_to_minimize)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    start_time = time.time()
    params = p_initial
    for i in range(NUM_STEPS_OPTAX_ADAM):
        opt_state, params = adam_step(i, opt_state, params)
    
    end_time = time.time()
    action_value_adam = action_to_minimize(params)
    return params, action_value_adam, end_time - start_time

def run_newtons_method(action_to_minimize, grad_action, hessian_action, p_initial):
    p_newton = p_initial
    start_time = time.time()
    for i in range(NUM_STEPS_NEWTON):
        grad_val = grad_action(p_newton)
        hess_val = hessian_action(p_newton)
        hess_val += jnp.eye(hess_val.shape[0]) * 1e-6  # Regularization
        delta_p = jnp.linalg.solve(hess_val, grad_val)
        p_newton = p_newton - delta_p
    end_time = time.time()
    action_value_newton = action_to_minimize(p_newton)
    return p_newton, action_value_newton, end_time - start_time

def run_hessian_optimization(action_to_minimize, grad_action, hessian_action, p_initial):
    p_hessian = p_initial
    start_time = time.time()
    for i in range(NUM_STEPS_HESSIAN):
        grad_val = grad_action(p_hessian)
        hess_val = hessian_action(p_hessian)
        hess_val += jnp.eye(hess_val.shape[0]) * 1e-6  # Regularization
        delta_p = jnp.linalg.solve(hess_val, grad_val)
        p_hessian = p_hessian - LEARNING_RATE_HESSIAN * delta_p
    end_time = time.time()
    action_value_hessian = action_to_minimize(p_hessian)
    return p_hessian, action_value_hessian, end_time - start_time

def run_yogi_optimization(action_to_minimize, p_initial):
    optimizer = optax.yogi(learning_rate=STEP_SIZE_YOGI)
    opt_state = optimizer.init(p_initial)
    
    @jax.jit
    def yogi_step(i, opt_state, params):
        grads = jax.grad(action_to_minimize)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    start_time = time.time()
    params = p_initial
    for i in range(NUM_STEPS_YOGI):
        opt_state, params = yogi_step(i, opt_state, params)
    
    end_time = time.time()
    action_value_yogi = action_to_minimize(params)
    return params, action_value_yogi, end_time - start_time

def run_lbfgs_optimization(action_to_minimize, p_initial, num_steps=NUM_STEPS_LBFGS, learning_rate=STEP_SIZE_LBFGS):
    # Initialize LBFGS optimizer with optional learning rate
    optimizer = optax.lbfgs(learning_rate=learning_rate)
    opt_state = optimizer.init(p_initial)
    
    @jax.jit
    def lbfgs_step(i, opt_state, params):
        # Obtain function value and gradients at the current parameters
        value = action_to_minimize(params)
        grads = jax.grad(action_to_minimize)(params)
        
        # Perform the LBFGS update step
        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=value, grad=grads, value_fn=action_to_minimize
        )
        # Apply updates to parameters
        params = optax.apply_updates(params, updates)
        return opt_state, params

    # Run the optimization for a specified number of steps
    start_time = time.time()
    params = p_initial
    for i in range(num_steps):  # Limit iterations with this loop
        opt_state, params = lbfgs_step(i, opt_state, params)
    
    end_time = time.time()
    action_value_lbfgs = action_to_minimize(params)
    return params, action_value_lbfgs, end_time - start_time

def run_adabelief_optimization(action_to_minimize, p_initial):
    optimizer = optax.adabelief(learning_rate=STEP_SIZE_ADABELIEF)
    opt_state = optimizer.init(p_initial)
    
    @jax.jit
    def adabelief_step(i, opt_state, params):
        grads = jax.grad(action_to_minimize)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    start_time = time.time()
    params = p_initial
    for i in range(NUM_STEPS_ADABELIEF):
        opt_state, params = adabelief_step(i, opt_state, params)
    
    end_time = time.time()
    action_value_adabelief = action_to_minimize(params)
    return params, action_value_adabelief, end_time - start_time

def print_final_comparison(bfgs_result, bfgs_time, adam_result, adam_time, yogi_result, yogi_time,
                           lbfgs_result, lbfgs_time, adabelief_result, adabelief_time,
                           newton_result, newton_time, hessian_result, hessian_time,
                           optax_adam_result, optax_adam_time):
    print("\nFinal Action Values and Time Comparison:")
    print(f"BFGS: {bfgs_result} | Time Taken: {bfgs_time:.4f} seconds")
    print(f"Adam (JAX): {adam_result} | Time Taken: {adam_time:.4f} seconds")
    print(f"Adam (Optax): {optax_adam_result} | Time Taken: {optax_adam_time:.4f} seconds")
    print(f"Yogi: {yogi_result} | Time Taken: {yogi_time:.4f} seconds")
    print(f"LBFGS: {lbfgs_result} | Time Taken: {lbfgs_time:.4f} seconds")
    print(f"AdaBelief: {adabelief_result} | Time Taken: {adabelief_time:.4f} seconds")
    print(f"Newton's Method: {newton_result} | Time Taken: {newton_time:.4f} seconds")
    print(f"Hessian-based Optimization: {hessian_result} | Time Taken: {hessian_time:.4f} seconds")
