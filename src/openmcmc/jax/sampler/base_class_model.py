"""Test baseclass model"""

from abc import ABC
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


# @register_pytree_node_class
# @dataclass
@partial(jax.tree_util.register_dataclass, data_fields=["q_factor"], meta_fields=[])
@dataclass
class ExampleModel(ABC):
    """Example model base class

    Attributes:
       mean (Union[str, Identity, LinearCombination, MixtureParameterVector]): mean parameter (of class Parameter).
       precision (Union[str, Identity, ScaledMatrix, MixtureParameterMatrix]): precision parameter (of class Parameter).

    """

    q_factor: jnp.ndarray

    def get_b_factor(self, state: jnp.array):
        """Get b factor"""
        # if jnp.sum(state) > 10:
        #     b_factor = jnp.array([1., 1.])
        # else:
        #     b_factor = jnp.array([3., 3.])

        b_factor = jnp.array([1.0, 1.0])

        return b_factor

    # def tree_flatten(self):
    #     children = self.q_factor
    #     aux_data = None
    #     return (children, aux_data)

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(*children, **aux_data)
