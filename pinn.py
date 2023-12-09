import jax 
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import List


N_SAMPLES = 200
LAYERS = [ 2,10,10,10,1]
LEARNING_RATE= 0.001
N_EPOCHS = 30_000

key = jax.random.PRNGKey(0)
