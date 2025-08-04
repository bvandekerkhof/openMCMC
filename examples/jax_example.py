"""Test example jax with pytrees"""

import datetime

import jax
from jax import jit
from jax import numpy as jnp
from jax import random
from tqdm import tqdm

from openmcmc.jax.sampler.base_class_model import ExampleModel
from openmcmc.jax.sampler.base_class_sampler import OneSampler

jax.config.update("jax_traceback_filtering", "off")


@jit
def run_samplers(sampler_list: list, state: dict, key: jax.Array):
    "Jax function to run samplers"
    for value in sampler_list:
        state, key = value.sample(state, key)

    return state, key


@jit
def cholesky_solve(a_array: jax.Array, b_array: jax.Array):
    "Jax function cholseky solve"
    chol_factor = jax.scipy.linalg.cho_factor(a_array)
    y_result = jax.scipy.linalg.cho_solve(chol_factor, b_array)
    return y_result


@jit
def jit_bicg(a_array: jax.Array, b_array: jax.Array):
    "Jax function BICG"
    y_result, _ = jax.scipy.sparse.linalg.bicgstab(a_array, b_array)
    return y_result


@jit
def jit_cg(a_array: jax.Array, b_array: jax.Array):
    "Jax function CG"
    y_result, _ = jax.scipy.sparse.linalg.cg(a_array, b_array)
    return y_result


@jit
def jit_gmres(a_array: jax.Array, b_array: jax.Array):
    "Jax function gmres"
    y_result, _ = jax.scipy.sparse.linalg.gmres(a_array, b_array)
    return y_result


def run_cases(A: jax.Array, b: jax.Array, nof_it: int):
    "wrapper to easily run the cases"
    # print("Normal Cholesky solve")
    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        cfac = jax.scipy.linalg.cho_factor(A)
        y_compare = jax.scipy.linalg.cho_solve(cfac, b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Normal Cholesky\t{elapsed_time.total_seconds()}")
    # print(y)

    # print("Jit Normal Cholesky solve")
    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = cholesky_solve(a_array=A, b_array=b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Jit cholesky\t{elapsed_time.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y, _ = jax.scipy.sparse.linalg.bicgstab(A, b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"sparse bicg\t{elapsed_time.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = jit_bicg(a_array=A, b_array=b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"jit bicg\t{elapsed_time.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("Sparse linalg CG solve")
    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y, _ = jax.scipy.sparse.linalg.cg(A, b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"sparse cg\t{elapsed_time.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("Jit Sparse linalg CG solve")
    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = jit_cg(a_array=A, b_array=b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"jit sparse cg\t{elapsed_time.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("Sparse linalg gmres solve")
    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y, _ = jax.scipy.sparse.linalg.gmres(A, b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"sparse gmres\t{elapsed_time.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("JIT Sparse linalg gmres solve")
    start_time = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = jit_gmres(a_array=A, b_array=b)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"jit gmres\t{elapsed_time.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")


if __name__ == "__main__":
    # run_case = "random_number"
    run_case = "cholesky"

    if run_case == "random_number":
        model_one = ExampleModel(q_factor=1)
        model_two = ExampleModel(q_factor=2)
        sampler_one = OneSampler(param="one", model=model_one)
        sampler_two = OneSampler(param="two", model=model_two)
        current_state = {"one": jnp.array([1.0, 0.0]), "two": jnp.array([0.0, 1.0])}
        current_key = random.key(39)

        print(f"Current State {current_state}\nCurrent key{current_key}")

        NOF_ITERATIONS = 10000

        for _ in tqdm(range(int(NOF_ITERATIONS / 2))):
            current_state, current_key = sampler_one.sample(current_state=current_state, current_key=current_key)
            current_state, current_key = sampler_two.sample(current_state=current_state, current_key=current_key)

        print(f"Current State {current_state}\nCurrent key{current_key}")

        model_one.q_factor = 5

        for _ in tqdm(range(int(NOF_ITERATIONS / 2))):
            current_state, current_key = sampler_one.sample(current_state=current_state, current_key=current_key)
            current_state, current_key = sampler_two.sample(current_state=current_state, current_key=current_key)

        print(current_state)

        model_one = ExampleModel(q_factor=1)
        model_two = ExampleModel(q_factor=2)
        sampler_one = OneSampler(param="one", model=model_one)
        sampler_two = OneSampler(param="two", model=model_two)
        current_state = {"one": jnp.array([1.0, 0.0]), "two": jnp.array([0.0, 1.0])}
        current_key = random.key(39)

        print(f"Current State {current_state}\nCurrent key{current_key}")

        for _ in tqdm(range(int(NOF_ITERATIONS / 2))):
            current_state, current_key = run_samplers(
                sampler_list=[sampler_one, sampler_two], state=current_state, key=current_key
            )

        print(f"Current State {current_state}\nCurrent key{current_key}")

        model_one.q_factor = 5

        for _ in tqdm(range(int(NOF_ITERATIONS / 2))):
            current_state, current_key = run_samplers(
                sampler_list=[sampler_one, sampler_two], state=current_state, key=current_key
            )
        print(current_state)
    elif run_case == "cholesky":

        NOF_ITERATIONS = 10

        A_input = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        b_input = jnp.array([3.0, 4.0])
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)

        factor = 3
        print(f"\nFactor {factor}")
        temp_A = factor * [A_input]
        A_input = jax.scipy.linalg.block_diag(*temp_A)
        b_input = jnp.tile(b_input, factor)
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)

        factor = 10
        print(f"\nFactor {factor}")
        temp_A = factor * [A_input]
        A_input = jax.scipy.linalg.block_diag(*temp_A)
        b_input = jnp.tile(b_input, factor)
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)

        factor = 100
        print(f"\nFactor {factor}")
        temp_A = factor * [A_input]
        A_input = jax.scipy.linalg.block_diag(*temp_A)
        b_input = jnp.tile(b_input, factor)
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)


# https://docs.jax.dev/en/latest/working-with-pytrees.html
# https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax/code#Pytrees
# https://docs.jax.dev/en/latest/pytrees.html#extending-pytrees

# NOTE Make sure we always output the state again. Do we care it is a dictionary where something is replaced?
#   It seems to work for now.
# NOTE Add some sparse matrix algebra on bigger vectors and matrices
# NOTE can we use vmap even more to speed up?
