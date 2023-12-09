import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from jax import vmap, jit
import jax
import numpy as np
import time
from utils import calculate_cfl
import matplotlib.pyplot as plt
from functools import partial

import optax 


@jit
def heat_eq_step(temperature, source,dt, heat_cond_coeff):
    Nx = temperature.shape[0]
    A_mat = jnp.diag(jnp.ones(Nx-1),1) - 2*jnp.diag(jnp.ones(Nx),0) + jnp.diag(jnp.ones(Nx-1),-1)
    A_mat.at[0,:].set(0)
    A_mat.at[-1,:].set(0)

    temperature = temperature +dt * heat_cond_coeff * A_mat @ temperature +dt * source


    return temperature

def main():
    
    Nx = 40
    Nt = 4000
    heat_cond_coeff = 0.1
    dt = 40./Nt
    dx = 2./(Nx-1)

    print("CFL: ", calculate_cfl(2./Nx, dt, heat_cond_coeff))

    x = jnp.linspace(-1,1,Nx)
    t = jnp.linspace(0,1,Nt)

    ref_source_function = lambda x : 100* jnp.exp(-x**2/0.3)
    reference_source = vmap(ref_source_function)(x) 
    

    #check jax.lax.scan
    def run_heat_simulation(source):
        temperatures = jnp.ones((Nt,Nx))

        def body_function(i,temperature_array, source, dt, heat_cond_coeff):
            temperature_array = temperature_array.at[i,:].set(heat_eq_step(temperature_array[i-1,:], source ,dt, heat_cond_coeff))
            return temperature_array
        
        partial_body_function = partial(body_function, source=source, dt=dt, heat_cond_coeff=heat_cond_coeff)

        temperatures = jax.lax.fori_loop(1, Nt, partial_body_function, temperatures)
        return temperatures

    start = time.time()

    reference_temperatures = run_heat_simulation(reference_source)

    def find_sensor_indices(x,sensor_positions):
        sensor_indices = []
        for sensor_position in sensor_positions:
            sensor_indices.append(np.argmin(np.abs(x-sensor_position)))
        return jnp.array(sensor_indices)
    
    sensor_positions = [-0.9,-0.6,-0.3,0,0.3,0.6,0.9]
    sensor_ind = find_sensor_indices(x,sensor_positions)
    reference_sensor_measurements = reference_temperatures[:,sensor_ind]
   

    @jit
    def loss_fun(source,sensor_ind,reference_sensor_measurements):
        temperature_array = run_heat_simulation(source)
        loss = 0.
        sim_sensor_measurements = temperature_array[:,sensor_ind]
        loss += jnp.linalg.norm(sim_sensor_measurements - reference_sensor_measurements)
        return loss
    

    learning_rate = 1
    schedule = optax.exponential_decay(learning_rate, transition_steps=100, decay_rate=0.5, staircase=True)
    optimizer = optax.adam(schedule)
    source =  jnp.ones_like(x)
    opt_state = optimizer.init(source)

    optim_step = 300
    for i in range(optim_step):
        loss,grad = jax.value_and_grad(loss_fun)(source,sensor_ind,reference_sensor_measurements) 
        print("Optimization step: ", i, "loss: ", loss)
        updates, opt_state = optimizer.update(grad, opt_state)
        source = optax.apply_updates(source, updates)
        

    end = time.time()
    print("Time elapsed: ", end-start)

    plt.plot(x,reference_source)
    plt.plot(x,source)
    plt.show()

if __name__ == "__main__":
    main()
