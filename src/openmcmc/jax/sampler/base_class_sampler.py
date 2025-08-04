"""Test baseclass sampler"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax import random
from functools import partial

from openmcmc.jax.sampler.base_class_model import ExampleModel


# @register_pytree_node_class
# @partial(jax.tree_util.register_dataclass,
#                    data_fields=['model'],
#                    meta_fields=['param'])
@dataclass
class BaseSampler(ABC):
    """Abstract base class for openMCMC sampling algorithms for a model parameter.

    Attributes:
        param (str): label of the parameter to be sampled.
        model (Model): sub-model of overall model, containing only distributions with some dependence on self.param.

    """

    param: str
    model: ExampleModel

    @abstractmethod
    def sample(self, current_state: dict, current_key: jax._src.prng.PRNGKeyArray) -> dict:
        """Generate the next sample in the chain.

        Args:
            current_state (dict): dictionary containing current parameter values.
            current_key (jax._src.prng.PRNGKeyArray): Current random key to use

        Returns:
            new_dtate (dict): Updated state
            new_key (jax._src.prng.PRNGKeyArray): New random number key to use


        """

    # def tree_flatten(self):
    #     # children = self.model.tree_flatten()
    #     # aux_data = {'param': self.param}
    #     # return (children, aux_data)


    # def tree_unflatten():
    #     # return cls(*children, **aux_data)


# @register_pytree_node_class
# @dataclass

@partial(jax.tree_util.register_dataclass,
                   data_fields=['model'],
                   meta_fields=['param'])
@dataclass
class OneSampler(BaseSampler):
    """Normal-Normal conditional sampling (exploiting conjugacy)."""

    def sample(self, current_state: dict, current_key: jax.Array) -> dict:
        """Generate a sample

        Args:
            current_state (dict): dictionary containing the current sampler state.

        Returns:
            (dict): state with updated value for self.param.

        """
        n_param = current_state[self.param].shape[0]
        Q = jnp.ones((n_param, n_param))
        b = jnp.zeros(shape=(n_param, 1))

        Q += self.model.q_factor
        b = Q @ self.model.get_b_factor(state=current_state[self.param])
        # print(f"param: {self.param}")
        # print(f"current state: {current_state}")
        # print(f"b: {b}")

        # Add the 1 to get a new key to work with later
        # current_key, *subkeys = random.split(current_key, n_param+1)
        subkeys = random.split(current_key, n_param+1)
        current_key = subkeys[0]
        subkeys = subkeys[1:]
        # TODO Not sure if we need to explicitely delete these keys?
        random_values = jax.vmap(random.normal)(subkeys)
        # print(random_values)
        # random_values = 0
        del subkeys
            
        current_state[self.param] = current_state[self.param] + b + random_values
        # new_state = current_state[self.param] + b + random_values
        # print(f"updated state: {current_state}")
        return current_state, current_key
    
    # def tree_flatten(self):
    #     children = None
    #     aux_data = None
    #     return (children, aux_data)

    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     return cls(*children)
    
    # def tree_flatten(self):
    #     children = self.model
    #     aux_data = {'param': self.param}
    #     return (children, aux_data)

    # def tree_unflatten(cls, aux_data, children):
    #     return cls(*children, **aux_data)
