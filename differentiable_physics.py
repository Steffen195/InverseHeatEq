import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from jax import vmap, jit
import jax
import numpy as np
import time

import matplotlib.pyplot as plt
from functools import partial
from finite_difference import HE_FDM
import optax 
from utils import calculate_cfl, plot_heatmap, plot_ref_and_test_heatmap
def main():

    number_of_sensors= 10
    Lx = 2.
    Nx = 50
    Lt = 50
    Nt = 50000
    heat_cond_coeff = 0.5

    ref_he_fdm = HE_FDM(Lx, Nx, Lt,Nt, heat_cond_coeff)
  
    source_function = lambda x,t : jnp.exp(-(x)**2/0.3)
    meshgrid = jnp.meshgrid(ref_he_fdm.x, ref_he_fdm.t)
    ref_source = source_function(meshgrid[0], meshgrid[1])

    T_0 = jnp.ones((Nx,))

    
    def find_sensor_indices(x,sensor_positions):
        sensor_indices = []
        for sensor_position in sensor_positions:
            sensor_indices.append(np.argmin(np.abs(x-sensor_position)))
        return jnp.array(sensor_indices)
    
    sensor_positions = jnp.linspace(-Lx/2*0.95,Lx/2*0.95,number_of_sensors)
    sensor_ind = find_sensor_indices(ref_he_fdm.x,sensor_positions)
    
   
    def sensor_rollout(stepper_fn):
        def scan_fn(T, source):
            T_next = stepper_fn(T,source)
            return T_next, T_next[sensor_ind]
        
        def sensor_rollout_fn(source):
            _, trj = jax.lax.scan(scan_fn,T_0,source,length=stepper_fn.Nt)
            return trj
        
        return sensor_rollout_fn
    
    def full_rollout(stepper_fn):
        def scan_fn(T, source):
            T_next = stepper_fn(T,source)
            return T_next, T_next
        
        def rollout_fn(source):
            _, trj = jax.lax.scan(scan_fn,T_0,source,length=stepper_fn.Nt)
            return trj
        
        return rollout_fn
    

    ref_sensor_trj = sensor_rollout(ref_he_fdm)(ref_source)

    @jit
    def loss_fun(source,ref_sensor_trj):
        curr_test_sensor_trj = sensor_rollout(ref_he_fdm)(source)
        loss = 0.
        loss += jnp.linalg.norm(ref_sensor_trj-curr_test_sensor_trj)
        return loss
    

    learning_rate = 0.01
    schedule = optax.exponential_decay(learning_rate, transition_steps=100, decay_rate=0.2, staircase=True)
    optimizer = optax.adam(schedule)
    source =  jnp.ones_like(ref_source)
    opt_state = optimizer.init(source)

    optim_step = 1000
    for i in range(optim_step):
        loss,grad = jax.value_and_grad(loss_fun)(source,ref_sensor_trj) 
        print("Optimization step: ", i, "loss: ", loss)
        updates, opt_state = optimizer.update(grad, opt_state)
        source = optax.apply_updates(source, updates)
        
    ref_trj = full_rollout(ref_he_fdm)(ref_source)
    test_trj = full_rollout(ref_he_fdm)(source)

    plot_ref_and_test_heatmap(ref_trj, test_trj,ref_he_fdm)
    plot_ref_and_test_heatmap(ref_source,source,ref_he_fdm)
    

if __name__ == "__main__":
    main()
