from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import jax
import jax.numpy as jnp

from utils import plot_heatmap, plot_diffusivity, plot_ref_and_test_heatmap
from functools import partial

import optax

import wandb
wandb.init(project="InverseHeat2")

Nx = 1000
Lx = 2 
Nt = 100
Lt = 1
dt = Lt/(Nt-1)
dx = Lx/(Nx-1)

r = dt/(2*dx**2)

optimstep = 200
learning_rate = 0.1
transition_steps = 50
decay_rate = 0.1
window_length = 30
number_sensors = 20
wandb.config = {"epochs": optimstep, "learning_rate": learning_rate, 
                "transition_steps":transition_steps, "decay_rate": decay_rate,
                "window_length": window_length,"number_sensors":number_sensors}



x = np.linspace(-Lx/2,Lx/2,Nx)
t = np.linspace(0,Lt,Nt)

T_0_function = lambda x: 0*x+1
T_0 = T_0_function(x)

ref_source_function = lambda x,t : np.exp(-(x)**2/0.3)
meshgrid = np.meshgrid(x, t)
ref_source = ref_source_function(meshgrid[0], meshgrid[1])

ref_diffusivity_function = lambda x: 1*x**2+0.1
ref_diffusivity = ref_diffusivity_function(x)


sensor_positions = np.linspace(-Lx/2*0.99,Lx/2*0.99,number_sensors)

def find_sensor_indices(x,sensor_positions):
    sensor_indices = []
    for sensor_position in sensor_positions:
        sensor_indices.append(np.argmin(np.abs(x-sensor_position)))
    return np.array(sensor_indices)
    
sensor_ind = find_sensor_indices(x,sensor_positions)
sensor_positions = x[sensor_ind]  


def create_diff_matrix(Nx):
    central_diff_matrix = 0.5*(np.diag(np.ones(Nx-1),1) - np.diag(np.ones(Nx-1),-1))
    #Incorporate constant boundary conditions by setting the first and last row to zero
    central_diff_matrix[0,:]= 0
    central_diff_matrix[-1,:]= 0
    return central_diff_matrix

def lhs_rhs_creation(diffusivity,Nx,r):
    central_diff_matrix = create_diff_matrix(Nx)

    lhs = np.eye(Nx) - r*(central_diff_matrix@jnp.diag(diffusivity)@central_diff_matrix)
    rhs = np.eye(Nx) + r*(central_diff_matrix@jnp.diag(diffusivity)@central_diff_matrix)

    return lhs, rhs


def create_step_fn(diffusivity):
    lhs, rhs = lhs_rhs_creation(diffusivity,Nx,r)
    def step_fn(temperature,source):
        temperature = jax.scipy.linalg.solve(lhs,rhs@temperature+dt*source)
        return temperature
    return step_fn


window = jnp.ones(window_length) / window_length
@jax.jit
def make_diffusivity_physical(diffusivity):
    diffusivity = jnp.abs(diffusivity)
    #diffusivity = jnp.convolve(diffusivity, window, mode='same')
    return diffusivity


def create_rollout_func(diffusivity):
    step_fn = create_step_fn(diffusivity)

    def scan_fn(T,source):
        T_next = step_fn(T,source)
        return T_next, T_next
    
    def rollout_fn(source):
        _, trj = jax.lax.scan(scan_fn,T_0,source,Nt)
        return trj

    return rollout_fn


def create_sensor_rollout_func(diffusivity):
    step_fn = create_step_fn(diffusivity)

    def scan_fn(T,source):
        T_next = step_fn(T,source)
        return T_next, T_next[sensor_ind]
    

    def rollout_fn(source):
        _, trj = jax.lax.scan(scan_fn,T_0,source,Nt)
        return trj

    return jax.jit(rollout_fn)


ref_trj = create_rollout_func(diffusivity=ref_diffusivity)(source=ref_source)
#plot_heatmap(ref_trj,Lt,Lx)
ref_sensor_trj = create_sensor_rollout_func(ref_diffusivity)(ref_source)

@jax.jit
def loss_fun(params,ref_sensor_trj):
    curr_test_sensor_trj = create_sensor_rollout_func(diffusivity=params[0])(source=params[1])
    loss = 0.
    loss += jnp.linalg.norm(ref_sensor_trj-curr_test_sensor_trj)
    return loss
    
#schedule = optax.exponential_decay(learning_rate, transition_steps= transition_steps, decay_rate=decay_rate, staircase=True)

optimizer = optax.adam(learning_rate=learning_rate)
source =  jnp.ones_like(ref_source)
diffusivity = 0.5*jnp.ones_like(ref_diffusivity)
params = [diffusivity,source]
opt_state = optimizer.init(params)

for i in range(optimstep):
    loss,grad = jax.value_and_grad(loss_fun)(params,ref_sensor_trj) 
    print("Optimization step: ", i, "loss: ", loss)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    params[0] = make_diffusivity_physical(params[0])
    wandb.log({"loss":loss})

final_test_trj = create_rollout_func(diffusivity=params[0])(source=params[1])
ref_trj = create_rollout_func(diffusivity=ref_diffusivity)(source=ref_source)

plot_diffusivity(ref_diffusivity,params[0],x,sensor_positions)
plot_ref_and_test_heatmap(ref_trj,final_test_trj,Lt,Lx)
plot_ref_and_test_heatmap(ref_source,params[1],Lt,Lx)
