"""Test example jax with pytrees"""

import datetime

import jax
from jax import jit
from jax import numpy as jnp
from jax import random
import jax.experimental.sparse as jsp
from tqdm import tqdm
import numpy as np
import scipy.sparse as sps

from openmcmc.jax.sampler.base_class_model import ExampleModel
from openmcmc.jax.sampler.base_class_sampler import OneSampler
from openmcmc.gmrf import sample_normal

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
    chol_FACTOR = jax.scipy.linalg.cho_factor(a_array)
    y_result = jax.scipy.linalg.cho_solve(chol_FACTOR, b_array)
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
    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        cfac = jax.scipy.linalg.cho_factor(A)
        y_compare = jax.scipy.linalg.cho_solve(cfac, b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"Normal Cholesky\t{elapsed_time_case.total_seconds()}")
    # print(y)

    # print("Jit Normal Cholesky solve")
    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = cholesky_solve(a_array=A, b_array=b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"Jit cholesky\t{elapsed_time_case.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y, _ = jax.scipy.sparse.linalg.bicgstab(A, b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"sparse bicg\t{elapsed_time_case.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = jit_bicg(a_array=A, b_array=b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"jit bicg\t{elapsed_time_case.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("Sparse linalg CG solve")
    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y, _ = jax.scipy.sparse.linalg.cg(A, b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"sparse cg\t{elapsed_time_case.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("Jit Sparse linalg CG solve")
    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = jit_cg(a_array=A, b_array=b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"jit sparse cg\t{elapsed_time_case.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("Sparse linalg gmres solve")
    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y, _ = jax.scipy.sparse.linalg.gmres(A, b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"sparse gmres\t{elapsed_time_case.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")

    # print("JIT Sparse linalg gmres solve")
    start_time_case = datetime.datetime.now()
    # for _ in tqdm(range(nof_it)):
    for _ in range(nof_it):
        y = jit_gmres(a_array=A, b_array=b)
    end_time_case = datetime.datetime.now()
    elapsed_time_case = end_time_case - start_time_case
    print(f"jit gmres\t{elapsed_time_case.total_seconds()}")
    if not jnp.allclose(y_compare, y):
        print("Ouput not close")


# NOTE tried to sparsify these functions but that does not work either.
@jsp.sparsify
def sparse_lu(Q):
    """Test sparsify"""
    _, L_matrix, U_matrix = jax.scipy.linalg.lu(a=Q)
    L = L_matrix.dot(jnp.diag(U_matrix.diagonal() ** 0.5))
    return L

@jit
def sample_normal_jit(mu, Q, random_key):
    """Jit compiled function for sample normal"""
    # NOTE the cholesky part is the slow bit, solve bit is slower for sparse matrices but okay.
    # Small differences in result seems to be due to floating point representation.
    # Can we invert Q and use sample multivariate normal -> no is slow
    # What happens when we convert csc matrix to jnp array -> does not help as cholesky function cannot handle a sparse jax array
    # Can we just call the regular regular scipy sparse function here and jit compile? -> NO, need to do it before and after depending if that helps or not, so just JIT compule inidividual components
    # Would we be able to do LUD decomposition quick instead -> Does not seem like it,
    #   also not able to sparsify jax functions from non-sparse module

    n_param = mu.size
    random_key, subkeys = random.split(random_key)
    # NOTE assuming the indexing is something jax does not like lets first
    # generate a new key and then the required subkeys from that one
    random_key, subkeys = random.split(random_key)
    subkeys = random.split(subkeys, n_param)
    z = jax.vmap(random.normal)(subkeys)
    del subkeys
    
    # _, L_matrix, U_matrix = jax.scipy.linalg.lu(a=Q)
    # L = L_matrix.dot(jnp.diag(U_matrix.diagonal() ** 0.5))

    fact_lu = sps.linalg.splu(Q, diag_pivot_thresh=0, options={"RowPerm": False, "ColPerm": False})
    L = fact_lu.L.dot(sps.diags(fact_lu.U.diagonal() ** 0.5))

    # L = sparse_lu(Q)

    # L = jax.scipy.linalg.cholesky(Q, lower=True)
    # result = jax.scipy.linalg.solve_triangular(L.T, z, trans=0, lower=False).reshape(z.shape) + mu
    # result = jax.scipy.linalg.solve_triangular(Q.T, z, trans=0, lower=False).reshape(z.shape) + mu

    result = L

    return result, random_key

if __name__ == "__main__":
    # RUN_CASE = "random_number"
    # RUN_CASE = "cholesky"
    RUN_CASE = "sample_normal"

    if RUN_CASE == "random_number":
        model_one = ExampleModel(q_FACTOR=1)
        model_two = ExampleModel(q_FACTOR=2)
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

        model_one.q_FACTOR = 5

        for _ in tqdm(range(int(NOF_ITERATIONS / 2))):
            current_state, current_key = sampler_one.sample(current_state=current_state, current_key=current_key)
            current_state, current_key = sampler_two.sample(current_state=current_state, current_key=current_key)

        print(current_state)

        model_one = ExampleModel(q_FACTOR=1)
        model_two = ExampleModel(q_FACTOR=2)
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

        model_one.q_FACTOR = 5

        for _ in tqdm(range(int(NOF_ITERATIONS / 2))):
            current_state, current_key = run_samplers(
                sampler_list=[sampler_one, sampler_two], state=current_state, key=current_key
            )
        print(current_state)

    elif RUN_CASE == "cholesky":

        NOF_ITERATIONS = 1000

        A_input = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        b_input = jnp.array([3.0, 4.0])
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)
        
        FACTOR = 3
        print(f"\nFACTOR {FACTOR}")
        temp_A = FACTOR * [A_input]
        A_input = jax.scipy.linalg.block_diag(*temp_A)
        b_input = jnp.tile(b_input, FACTOR)
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)

        FACTOR = 10
        print(f"\nFACTOR {FACTOR}")
        temp_A = FACTOR * [A_input]
        A_input = jax.scipy.linalg.block_diag(*temp_A)
        b_input = jnp.tile(b_input, FACTOR)
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)

        FACTOR = 500
        print(f"\nFACTOR {FACTOR}")
        temp_A = FACTOR * [A_input]
        A_input = jax.scipy.linalg.block_diag(*temp_A)
        b_input = jnp.tile(b_input, FACTOR)
        run_cases(A=A_input, b=b_input, nof_it=NOF_ITERATIONS)

    elif RUN_CASE == "sample_normal":
        NOF_ITERATIONS = 10000
        # for n_param in [10, 100, 1000]:
        for n_param in [10, 20, 50, 100, 500, 1000]:
        # for n_param in [10]:
            print(f"Number of parameters {n_param}")
            current_key = random.key(39)

            identity_matrix = np.identity(n_param)
            d_matrix = np.diff(identity_matrix, axis=0)
            d_matrix = d_matrix.T @ d_matrix
            d_matrix[0, 0] = 2
            d_matrix_sparse = sps.csc_matrix(d_matrix)
            d_matrix_jax = jnp.array(d_matrix)
            d_matrix_sparse_jax = jsp.CSC.fromdense(d_matrix)

            mu_vector = np.zeros((n_param, 1))
            # mu_vector_sparse = sps.csc_matrix(mu_vector)
            mu_vector_jax = jnp.array(mu_vector)
            # mu_vector_sparse_jax = jsp.CSC.fromdense(mu_vector)

            y_check, z_used, L_calculated = sample_normal(mu=mu_vector, Q=d_matrix, n=1)
            y_check_sparse, z_used_sparse, L_calculated_sparse = sample_normal(mu=mu_vector_jax, Q=d_matrix_sparse, n=1)
            
            z_jax = jnp.array(z_used)
            z_jax_from_sparse = jnp.array(z_used_sparse)
            L_jax = jax.scipy.linalg.cholesky(d_matrix_jax, lower=True)
            L_jax_sparse = jsp.CSC.fromdense(L_calculated)
            
            fact_P, fact_L, fact_U = jax.scipy.linalg.lu(a=d_matrix_jax)
            L_jax_2 = fact_L.dot(jnp.diag(fact_U.diagonal() ** 0.5))
            print(f"Lower cholesky jax equal? {np.allclose(L_calculated, L_jax)}")
            print(f"Lower cholesky jax 2 equal? {np.allclose(L_calculated, L_jax_2)}")
            # print(f"Lower cholesky jax sparse equal? {np.allclose(L_calculated, L_jax_sparse)}")

            y_jax = jax.scipy.linalg.solve_triangular(L_jax.T, z_jax, trans=0, lower=False)
            # This sparse solver seems to work, but TODO check speed and not sure if it is worth our while if we cannot jit compile the cholesky factorization at reasonable speed.
            y_jax_sparse = jax.experimental.sparse.linalg.spsolve(data=L_jax_sparse.data,
                                                                   indices=L_jax_sparse.indices,
                                                                   indptr=L_jax_sparse.indptr,
                                                                   b=z_jax_from_sparse.reshape(-1),
                                                                   tol=1e-06, reorder=1)
            print(f"jax solve equal? {np.allclose(y_check, y_jax)}")
            print(f"jax sparse solve equal? {np.allclose(y_check, y_jax_sparse)}")

            print("Starting iterations sample_normal")
            start_time = datetime.datetime.now()
            for _ in range(NOF_ITERATIONS):
                out_matrix, z_used, L_calculated = sample_normal(mu=mu_vector, Q=d_matrix, n=1)
                # out_matrix, z_used, L_calculated = sample_normal(mu=mu_vector, Q=d_matrix, n=1, L=L_original)
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            print(f"Elapsed \t{elapsed_time.total_seconds()}")
            
            print("Starting iterations sample_normal sparse")
            start_time = datetime.datetime.now()
            for _ in range(NOF_ITERATIONS):
                out_matrix_sparse = sample_normal(mu=mu_vector, Q=d_matrix_sparse, n=1)
                # out_matrix_sparse = sample_normal(mu=mu_vector_sparse, Q=d_matrix_sparse, n=1, L=L_original_sparse)
            
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            print(f"Elapsed \t{elapsed_time.total_seconds()}")
            
            print("Starting iterations sample_normal jit")
            start_time = datetime.datetime.now()
            for _ in range(NOF_ITERATIONS):
                y_jax, current_key = sample_normal_jit(mu=mu_vector_jax, Q=d_matrix_jax, random_key=current_key)
                # y_jax, current_key = sample_normal_jit(mu=mu_vector_jax, Q=L_jax, random_key=current_key)
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            print(f"Elapsed \t{elapsed_time.total_seconds()}")

            print("Starting iterations sample_normal jit sparsify")
            start_time = datetime.datetime.now()
            for _ in range(NOF_ITERATIONS):
                y_jax, current_key = sample_normal_jit(mu=mu_vector_jax, Q=d_matrix_sparse_jax, random_key=current_key)
                # y_jax, current_key = sample_normal_jit(mu=mu_vector_jax, Q=L_jax, random_key=current_key)
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            print(f"Elapsed \t{elapsed_time.total_seconds()}")

# https://docs.jax.dev/en/latest/working-with-pytrees.html
# https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax/code#Pytrees
# https://docs.jax.dev/en/latest/pytrees.html#extending-pytrees

# NOTE https://dansblog.netlify.app/posts/2022-11-27-sparse7/sparse7.html Seems like this is very slow. Probably not worth it?
# NOTE I think because the sparse LU decompozition from scipy is based on SuperLU which is already compiled C code we cannot beat it.
# TODO jit compile inner parts when not sparse matrix or when nof parameters is small enough?
