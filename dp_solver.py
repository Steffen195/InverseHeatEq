from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import jax
import jax.numpy as jnp

from utils import plot_heatmap, plot_ref_and_test_heatmap, plot_ref_initial_and_test_heatmap
from functools import partial

import optax

Nx = 200
Lx = 2
Nt = 200
Lt = 1
dt = Lt/(Nt-1)
dx = Lx/(Nx-1)
diffusivity = 0.5


r = dt/(2*dx**2)*diffusivity

optimstep = 500
learning_rate = 0.005
#transition_steps = 500
#decay_rate = 0.2
number_sensors = 20

W_and_B_Tracking = False

if W_and_B_Tracking== True:
    import wandb
    wandb.init(project="InverseHeat2")
    wandb.config = {"epochs": optimstep, "learning_rate": learning_rate, 
                #"transition_steps":transition_steps, "decay_rate": decay_rate,
                "number_sensors":number_sensors}

from jax import random
key = random.PRNGKey(0)


x = np.linspace(-Lx/2,Lx/2,Nx)
t = np.linspace(0,Lt,Nt)         

T_0_function = lambda x: 0*x+1
T_0 = T_0_function(x)

test_case = "Gaussian"
if test_case == "Constant":
    ref_source_function = lambda x,t : 3+0*x
elif test_case == "Piecewise":
    ref_source_function = lambda x,t: 1. if(x>-0.3 and x<0.3) else 0. 
elif test_case == "Gaussian":
    ref_source_function = lambda x,t: np.exp(-(x-0.25*np.sin(t*4))**2/0.3)
else:
    raise(ValueError("Please input a valid test case name"))

meshgrid = np.meshgrid(x, t)
ref_source = np.vectorize(ref_source_function)(meshgrid[0],meshgrid[1])

initial_source= jnp.ones_like(ref_source)
source = initial_source

sensor_positions = np.linspace(x[0],x[-1],number_sensors)

def find_sensor_indices(x,sensor_positions):
    sensor_indices = []
    for sensor_position in sensor_positions:
        sensor_indices.append(np.argmin(np.abs(x-sensor_position)))
    return np.array(sensor_indices)
    
sensor_ind = find_sensor_indices(x,sensor_positions)
sensor_positions = x[sensor_ind]  


def create_2nd_der_matrix(Nx):
    central_diff_matrix = (np.diag(np.ones(Nx-1),1) - 2*np.diag(np.ones(Nx),0)+ np.diag(np.ones(Nx-1),-1))
    #Incorporate constant boundary conditions by setting the first and last row to zero
    central_diff_matrix[0,:]= 0
    central_diff_matrix[-1,:]= 0
    return central_diff_matrix

def CN2_lhs_rhs_creation(Nx):
    central_diff_matrix = create_2nd_der_matrix(Nx)

    lhs = np.eye(Nx) - r*(central_diff_matrix)
    rhs = np.eye(Nx) + r*(central_diff_matrix)

    return lhs, rhs

def create_step_fn():
    lhs, rhs = CN2_lhs_rhs_creation(Nx)
    def step_fn(temperature,source):        
        temperature = jax.scipy.linalg.solve(lhs,rhs@temperature+dt*source)
        return temperature
    return step_fn


def create_rollout_func():
    step_fn = create_step_fn()

    def scan_fn(T,source):
        T_next = step_fn(T,source)
        return T_next, T_next
    
    def rollout_fn(source):
        _, trj = jax.lax.scan(scan_fn,T_0,source,Nt)
        return trj

    return rollout_fn

def create_sensor_rollout_func():
    step_fn = create_step_fn()

    def scan_fn(T,source):
        T_next = step_fn(T,source)
        return T_next, T_next[sensor_ind]
    
    def rollout_fn(source):
        _, trj = jax.lax.scan(scan_fn,T_0,source,Nt)
        return trj
    return jax.jit(rollout_fn)

rollout_fn = create_rollout_func()
sensor_rollout_fn = create_sensor_rollout_func()
ref_trj = rollout_fn(source=ref_source)
plot_heatmap(ref_trj,Lt,Lx)
ref_sensor_trj = sensor_rollout_fn(ref_source)

def loss_fun(params,ref_sensor_trj):
    curr_test_sensor_trj = sensor_rollout_fn(params[0])
    loss = 0.
    loss += jnp.linalg.norm(ref_sensor_trj-curr_test_sensor_trj)
    return loss


#schedule = optax.exponential_decay(learning_rate, transition_steps= transition_steps, decay_rate=decay_rate, staircase=True)

optimizer = optax.adam(learning_rate=learning_rate)

initial_params = [source]
params = initial_params
opt_state = optimizer.init(params)

for i in range(optimstep):
    loss,grad = jax.value_and_grad(loss_fun)(params,ref_sensor_trj) 
    print("Optimization step: ", i, "loss: ", loss)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)



initial_test_trj = rollout_fn(initial_source)
final_test_trj = rollout_fn(source=params[0])

temperature_plot = plot_ref_and_test_heatmap(ref_trj,final_test_trj,Lt,Lx,x,test_case)
source_plot = plot_ref_and_test_heatmap(ref_source,params[0],Lt,Lx,x,test_case)
source_plot.savefig(f"./figures/{test_case}_source.png")
temperature_plot.savefig(f"./figures/{test_case}_temperature.png")

initial_source_loss = jnp.linalg.norm(initial_source-ref_source)
final_source_loss = jnp.linalg.norm(params[0]-ref_source)

initial_temperature_loss = jnp.linalg.norm(initial_test_trj-ref_trj)
final_temperature_loss  = jnp.linalg.norm(final_test_trj - ref_trj)

print(f"Loss for source. Epoch 0: {initial_source_loss}, Final Epoch {final_source_loss}")
print(f"Loss for temperature. Epoch 0: {initial_temperature_loss}, Final Epoch {final_temperature_loss}")
