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


class HeatTransferProblem(): 

    def __init__(self):
        self.Nx = 40
        self.Nt = 4000
        self.heat_cond_coeff = 0.1
        self.dt = 40./self.Nt
        self.dx = 2./(self.Nx-1)

        self.x = jnp.linspace(-1,1,self.Nx)
        self.t = jnp.linspace(0,1,self.Nt)

    

    def define_source(self, source):
        self.source = source

    def run_heat_simulation(self):
        temperatures = jnp.ones((self.Nt,self.Nx))

        def body_function(i,temperature_array, source, dt, heat_cond_coeff):
            temperature_array = temperature_array.at[i,:].set(heat_eq_step(temperature_array[i-1,:], source ,dt, heat_cond_coeff))
            return temperature_array
        
        partial_body_function = partial(body_function, source=self.source, dt=self.dt, heat_cond_coeff=self.heat_cond_coeff)

        temperatures = jax.lax.fori_loop(1, self.Nt, partial_body_function, temperatures)
        return temperatures
    

def heat_eq_step(temperature, source,dt, heat_cond_coeff):
    Nx = temperature.shape[0]
    A_mat = jnp.diag(jnp.ones(Nx-1),1) - 2*jnp.diag(jnp.ones(Nx),0) + jnp.diag(jnp.ones(Nx-1),-1)
    A_mat.at[0,:].set(0)
    A_mat.at[-1,:].set(0)
    temperature = temperature +dt *heat_cond_coeff * A_mat @ temperature +dt * source
    return temperature

def main():
    
    heat_transfer_problem = HeatTransferProblem()
    source_function = lambda x : 100* jnp.exp(-x**2/0.3)
    heat_transfer_problem.define_source(vmap(source_function)(heat_transfer_problem.x))

    print("CFL: ", calculate_cfl(2./heat_transfer_problem.Nx, heat_transfer_problem.dt, heat_transfer_problem.heat_cond_coeff))

    start = time.time()

    reference_temperatures = heat_transfer_problem.run_heat_simulation()

    end = time.time()

    print("time: ", end-start)


if __name__ == "__main__":
    main()
