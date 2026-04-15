import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

from microQPSWIFT import microQPSWIFTSolver
from tester import Tester


P = jnp.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0.0, 0, 0, 1.0]])
p = jnp.array([1.0, -2.0, 1.0, 2.0])

A_eq = jnp.array([[1, 1, 0, 0], [1, 0, -1, 0]])
b_eq = jnp.array([8.0, -7.0])

G_ineq = jnp.array(
    [
        [1.0, 0, 0, 0],
        [-1, 0.0, 0, 0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
h_ineq = jnp.array([15.5, -15, 15.0])

tester = Tester(P, p, A_eq, b_eq, G_ineq, h_ineq)
tester.compare_solutions()
