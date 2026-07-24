"""barrierQP: a tiny JAX Mehrotra predictor-corrector QP solver.

The public entry point is :class:`Solver`; :func:`Solver.solve` returns a
:class:`Result` carrying the solution, status, and factor/solve counts.
"""

from .solver import Solver, Result

__all__ = ["Solver", "Result"]
