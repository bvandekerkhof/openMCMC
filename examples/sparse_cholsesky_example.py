from functools import partial
from scipy import sparse
import scipy as sp
from jax import lax
from jax import numpy as jnp
from jax import jit, grad
import numpy as np
import timeit
import datetime
import scipy.sparse as sps


def make_matrix(n):
    one_d = sparse.diags([[-1.0] * (n - 2), [2.0] * n, [-1.0] * (n - 2)], [-2, 0, 2])
    A = sparse.kronsum(one_d, one_d) + sparse.eye(n * n)
    A_csc = A.tocsc()
    A_csc.eliminate_zeros()
    A_lower = sparse.tril(A_csc, format="csc")
    A_index = A_lower.indices
    A_indptr = A_lower.indptr
    A_x = A_lower.data
    return (A_index, A_indptr, A_x, A_csc)


def _symbolic_factor(A_indices, A_indptr):
    # Assumes A_indices and A_indptr index the lower triangle of $A$ ONLY.
    n = len(A_indptr) - 1
    L_sym = [np.array([], dtype=int) for j in range(n)]
    children = [np.array([], dtype=int) for j in range(n)]

    for j in range(n):
        L_sym[j] = A_indices[A_indptr[j] : A_indptr[j + 1]]
        for child in children[j]:
            tmp = L_sym[child][L_sym[child] > j]
            L_sym[j] = np.unique(np.append(L_sym[j], tmp))
        if len(L_sym[j]) > 1:
            p = L_sym[j][1]
            children[p] = np.append(children[p], j)

    L_indptr = np.zeros(n + 1, dtype=int)
    L_indptr[1:] = np.cumsum([len(x) for x in L_sym])
    L_indices = np.concatenate(L_sym)

    return L_indices, L_indptr


def symbolic_cholesky3(A_indices, A_indptr, L_indptr, parent, nnz):
    @partial(jit, static_argnums=(4,))
    def _inner(A_indices_, A_indptr_, L_indptr, parent, nnz):
        ## Make everything a jnp array. Really should use jaxtyping
        A_indices_ = jnp.array(A_indices_)
        A_indptr_ = jnp.array(A_indptr_)
        L_indptr = jnp.array(L_indptr)
        parent = jnp.array(parent)

        ## innermost while loop
        def body_while(val):
            index, i, counter, node, mark = val
            mark = mark.at[node].set(i)
            index[0] = index[0].at[counter].set(node)  # column
            index[1] = index[1].at[counter].set(i)  # row
            return (index, i, counter + 1, parent[node], mark)

        def cond_while(val):
            index, i, counter, node, mark = val
            return lax.bitwise_and(lax.lt(node, i), lax.ne(mark[node], i))

        ## Inner for loop
        def body_inner_for(indptr, val):
            index, i, counter, mark = val
            node = A_indices_[indptr]

            index, i, counter, node, mark = lax.while_loop(cond_while, body_while, (index, i, counter, node, mark))
            return (index, i, counter, mark)

        ## Outer for loop
        def body_out_for(i, val):
            index, counter, mark = val
            mark = mark.at[i].set(i)
            index[0] = index[0].at[counter].set(i)
            index[1] = index[1].at[counter].set(i)
            counter = counter + 1
            index, i, counter, mark = lax.fori_loop(
                A_indptr_[i], A_indptr_[i + 1], body_inner_for, (index, i, counter, mark)
            )

            return (index, counter, mark)

        ## Body of code
        n = len(A_indptr_) - 1
        mark = jnp.repeat(-1, n)

        index = [jnp.zeros(nnz, dtype=int), jnp.zeros(nnz, dtype=int)]
        counter = 0

        init = (index, counter, mark)
        index, counter, mark = lax.fori_loop(0, n, body_out_for, init)

        return index

    n = len(A_indptr) - 1
    index = _inner(A_indices, A_indptr, L_indptr, parent, nnz)
    ## return jnp.lexsort[index[1][jnp.lexsort((index[1], index[0]))
    return sparse.coo_array((np.ones(nnz), (index[1], index[0])), shape=(n, n)).tocsc().indices


@jit
def etree(A_indices, A_indptr):
    # print("(Re-)compiling etree(A_indices, A_indptr)")
    ## innermost while loop
    def body_while(val):
        #  print(val)
        j, node, parent, col_count, row_count, mark = val
        update_parent = lambda x: x[0].at[x[1]].set(x[2])
        parent = lax.cond(lax.eq(parent[node], -1), update_parent, lambda x: x[0], (parent, node, j))
        mark = mark.at[node].set(j)
        col_count = col_count.at[node].add(1)
        row_count = row_count.at[j].add(1)
        return (j, parent[node], parent, col_count, row_count, mark)

    def cond_while(val):
        j, node, parent, col_count, row_count, mark = val
        return lax.bitwise_and(lax.lt(node, j), lax.ne(mark[node], j))

    ## Inner for loop
    def body_inner_for(indptr, val):
        j, A_indices, A_indptr, parent, col_count, row_count, mark = val
        node = A_indices[indptr]
        j, node, parent, col_count, row_count, mark = lax.while_loop(
            cond_while, body_while, (j, node, parent, col_count, row_count, mark)
        )
        return (j, A_indices, A_indptr, parent, col_count, row_count, mark)

    ## Outer for loop
    def body_out_for(j, val):
        A_indices, A_indptr, parent, col_count, row_count, mark = val
        mark = mark.at[j].set(j)
        j, A_indices, A_indptr, parent, col_count, row_count, mark = lax.fori_loop(
            A_indptr[j], A_indptr[j + 1], body_inner_for, (j, A_indices, A_indptr, parent, col_count, row_count, mark)
        )
        return (A_indices, A_indptr, parent, col_count, row_count, mark)

    ## Body of code
    n = len(A_indptr) - 1
    parent = jnp.repeat(-1, n)
    mark = jnp.repeat(-1, n)
    col_count = jnp.repeat(1, n)
    row_count = jnp.repeat(1, n)
    init = (A_indices, A_indptr, parent, col_count, row_count, mark)
    A_indices, A_indptr, parent, col_count, row_count, mark = lax.fori_loop(0, n, body_out_for, init)
    return (parent, col_count, row_count)


def dense_left_cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        L[j, j] = np.sqrt(A[j, j] - np.inner(L[j, :j], L[j, :j]))
        L[(j + 1) :, j] = (A[(j + 1) :, j] - L[(j + 1) :, :j] @ L[j, :j].transpose()) / L[j, j]
    return L

def dense_left_cholesky_two(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n-1):
        L[j, j] = np.sqrt(A[j, j] - np.inner(L[j, :j], L[j, :j]))
        L[(j + 1) :, j] = (A[(j + 1) :, j] - L[(j + 1) :, :j] @ L[j, :j].transpose()) / L[j, j]

    j=n-1
    L[j, j] = np.sqrt(A[j, j] - np.inner(L[j, :j], L[j, :j]))
    return L


@jit
def dense_left_cholesky_jit_two(A):
    n = A.shape[0]
    L = jnp.zeros_like(A)
    for j in range(n-1):
        L = L.at[j, j].set(jnp.sqrt(A[j, j] - jnp.inner(L[j, :j], L[j, :j])))
        L = L.at[(j + 1) :, j].set((A[(j + 1) :, j] - L[(j + 1) :, :j] @ L[j, :j].transpose()) / L[j, j])

    j=n-1
    L = L.at[j, j].set(jnp.sqrt(A[j, j] - jnp.inner(L[j, :j], L[j, :j])))
    return L


# TODO check if we can understand this article: https://dansblog.netlify.app/posts/2022-11-27-sparse7/sparse7.html#building-a-jax-traceable-symbolic-sparse-choleksy-factorisation
# TODO check if we can make the thing below work, maybe with vmap?
@jit
def dense_left_cholesky_jit(A):
    """Jit attemp cholesky""" 
    n = A.shape[0]
    L = jnp.zeros_like(A)
    for j in range(n):
        diag_A = A.at[j, j].get()
        L_row = L.at[j, :j].get()
        diag_L = jnp.sqrt(diag_A - jnp.inner(L_row, L_row))
        L = L.at[j, j].set(diag_L)
        next_row_A = A.at[(j + 1):, j].get(mode='fill', fill_value=jnp.nan)
        next_row_L = L.at[(j + 1):, :j].get(mode='fill', fill_value=jnp.nan)
        new_value = (next_row_A - next_row_L @ L_row.transpose()) / diag_L
        L = L.at[(j + 1):, j].set(new_value)
    return L


if __name__ == "__main__":
        # NOF_ITERATIONS = 10000
        # for n_param in [10, 20, 50, 100, 500, 1000]:
        NOF_ITERATIONS = 100
        for n_param in [10]:
            print(f"Number of parameters {n_param}")
            identity_matrix = np.identity(n_param)
            d_matrix = np.diff(identity_matrix, axis=0)
            d_matrix = d_matrix.T @ d_matrix
            d_matrix[0, 0] = 2
            d_matrix_jax = jnp.array(d_matrix)
            d_matrix_sparse = sps.csc_matrix(d_matrix)

            real_L = np.linalg.cholesky(d_matrix, upper=False)

            # print("Starting iterations L1")
            # start_time = datetime.datetime.now()
            # for _ in range(NOF_ITERATIONS):
            #     L_1 = dense_left_cholesky(d_matrix)    
            # end_time = datetime.datetime.now()
            # elapsed_time = end_time - start_time
            # diff_L_1 = np.sum(np.abs(real_L - L_1), axis=None)
            # print(f"Elapsed \t{elapsed_time.total_seconds():.4f}, difference: {diff_L_1}")

            print("Starting iterations L1 jax")
            start_time = datetime.datetime.now()
            for _ in range(NOF_ITERATIONS):
                L_1_jax = dense_left_cholesky_jit(d_matrix_jax)    
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            diff_L_1_jax = np.sum(np.abs(real_L - L_1_jax), axis=None)
            print(f"Elapsed \t{elapsed_time.total_seconds():.4f}, difference: {diff_L_1_jax}")

            # print("Starting iterations L2")
            # start_time = datetime.datetime.now()
            # for _ in range(NOF_ITERATIONS):
            #     L_2 = dense_left_cholesky_two(d_matrix)
            # end_time = datetime.datetime.now()
            # elapsed_time = end_time - start_time
            # diff_L_2 = np.sum(np.abs(real_L - L_2), axis=None)
            # print(f"Elapsed \t{elapsed_time.total_seconds():.4f}, difference: {diff_L_2}")

            # print("Starting iterations L2 jax")
            # start_time = datetime.datetime.now()
            # for _ in range(NOF_ITERATIONS):
            #     L_2_jax = dense_left_cholesky_jit_two(d_matrix_jax)
            # end_time = datetime.datetime.now()
            # elapsed_time = end_time - start_time
            # diff_L_2_jax = np.sum(np.abs(real_L - L_2_jax), axis=None)
            # print(f"Elapsed \t{elapsed_time.total_seconds():.4f}, difference: {diff_L_2_jax}")

            print("Starting iterations symbolic cholesky")
            start_time = datetime.datetime.now()
            parent, col_count, row_count = etree(d_matrix_sparse.indices, d_matrix_sparse.indptr)
            L_indptr = np.zeros(d_matrix_sparse.shape[0] + 1, dtype=int)
            L_indptr[1:] = np.cumsum(col_count)
            for _ in range(NOF_ITERATIONS):

                L_indices = symbolic_cholesky3(d_matrix_sparse.indices, d_matrix_sparse.indptr, L_indptr, parent, nnz=int(L_indptr[-1]))

            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            # diff_L_2_jax = np.sum(np.abs(real_L - L_2_jax), axis=None)
            print(f"Elapsed \t{elapsed_time.total_seconds():.4f}")

    # A_indices, A_indptr, A_x, A = make_matrix(15)
    # parent, col_count, row_count = etree(A.indices, A.indptr)
    # L_indptr = np.zeros(A.shape[0] + 1, dtype=int)
    # L_indptr[1:] = np.cumsum(col_count)
    # L_indices = symbolic_cholesky3(A.indices, A.indptr, L_indptr, parent, nnz=int(L_indptr[-1]))
    # L_indices_true, L_indptr_true = _symbolic_factor(A_indices, A_indptr)
    # print(all(L_indices == L_indices_true))
    # print(all(L_indptr == L_indptr_true))

    # L_test_dense = L_indices.to_dense()
    # A_indices, A_indptr, A_x, A = make_matrix(15)
    # parent, col_count, row_count = etree(A.indices, A.indptr)
    # L_indptr = np.zeros(A.shape[0] + 1, dtype=int)
    # L_indptr[1:] = np.cumsum(col_count)
    # L_indices = symbolic_cholesky3(A.indices, A.indptr, L_indptr, parent, nnz=int(L_indptr[-1]))
    # L_indices_true, L_indptr_true = _symbolic_factor(A_indices, A_indptr)
    # print(all(L_indices == L_indices_true))
    # print(all(L_indptr == L_indptr_true))


    # NOTE Assuming we cannot speed this up asand this symbolic_cholesky3 function is already too slow, let alone finding the actual values for the L matrix
