import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from jax import vmap, jit
import jax
import numpy as np
import time
from utils import calculate_cfl, plot_heatmap
import matplotlib.pyplot as plt
from functools import partial

import optax 


class HE_FDM(): 

    def __init__(self, Lx, Nx, Lt, Nt , heat_cond_coeff):
        self.Nx = Nx
        self.Nt = Nt
        self.Lx = Lx
        self.Lt = Lt
        self.heat_cond_coeff = heat_cond_coeff
        self.dt = Lt/Nt
        self.dx = Lx/Nx
        self.x = jnp.linspace(-Lx/2,Lx/2,Nx)
        self.t = jnp.linspace(0,Lt,Nt)
        self.A_mat =  self.set_A_mat()

        try:
            cfl = calculate_cfl(self.dx, self.dt, self.heat_cond_coeff)
            assert  cfl < 0.5
        except AssertionError:
            print(f"CFL condition not satisfied: CFL = {cfl}")
            exit()

    def set_A_mat(self):
        A_mat = jnp.diag(jnp.ones(self.Nx-1),1) - 2*jnp.diag(jnp.ones( self.Nx),0) + jnp.diag(jnp.ones(self.Nx-1),-1)
        A_mat.at[0,:].set(0)
        A_mat.at[-1,:].set(0)
        return A_mat

    def define_source(self, source):
        self.source = source

    def __call__(self, temperature,source):
        temperature = temperature +self.dt *self.heat_cond_coeff * self.A_mat @ temperature +self.dt * source
        return temperature
        
def main():
    
    Lx = 2.
    Nx = 100
    Lt = 20
    Nt = 11000
    heat_cond_coeff = 0.1

    ref_he_fdm = HE_FDM(Lx, Nx, Lt,Nt, heat_cond_coeff)
  
    source_function = lambda x,t : jnp.exp(-(x+(t-1)/20)**2/0.3)
    meshgrid = jnp.meshgrid(ref_he_fdm.x, ref_he_fdm.t)
    source = source_function(meshgrid[0], meshgrid[1])
    plot_heatmap(source, ref_he_fdm)
    T_0 = jnp.ones((Nx,))

    def rollout(stepper_fn):
        def scan_fn(T, source):
            T_next = stepper_fn(T,source)
            return T_next, T_next
        
        def rollout_fn(source):
            _, trj = jax.lax.scan(scan_fn,T_0,source,length=stepper_fn.Nt)
            return trj
        
        return rollout_fn
    
    start = time.time()
    trj = rollout(ref_he_fdm)(source)
    end = time.time()
    print("time: ", end-start)

    plot_heatmap(trj, ref_he_fdm)

if __name__ == "__main__":
    main()
