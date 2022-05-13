# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from tvm.script import tir as T


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    with T.iter([I, K, J], "SSR", "csrmm") as [vi, vk, vj]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def csrmm_dense_iter(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    with T.iter([I, J_detach, K], "SRS", "csrmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def segment_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    n: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    J = T.dense_variable(I, (100, nnz), indptr, "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")
    with T.iter([I, J], "SR", "segment_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def csr_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")
    with T.iter([I, J], "SR", "csr_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    J_detach = T.dense_fixed(mb)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, BI, BJ, F, J], "SSRSR", "bsrmm") as [
        vi,
        vbi,
        vbj,
        vf,
        vj,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def ellmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    feat_size: T.int32,
    col: T.int32,
    blk: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(nb)
    J = T.sparse_fixed(I, (mb, col), indices, "int32")
    J_detach = T.dense_fixed(mb)
    F = T.dense_fixed(feat_size)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, J, BI, BJ, F], "SRSRS", "ellmm") as [
        vi,
        vj,
        vbi,
        vbj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def csr_element_wise(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I, J), "float32")

    with T.iter([I, J], "SS", "csr_element_wise") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.5


@T.prim_func
def hyper_gnn(
    x: T.handle,
    y: T.handle,
    indptr: T.handle,
    indices: T.handle,
    indptr_T: T.handle,
    indices_T: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
    feat_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    F = T.dense_fixed(feat_size)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(m)
    I_T = T.sparse_variable(J_detach, (n, nnz), (indptr_T, indices_T), "int32")
    X = T.match_sparse_buffer(x, (I, F), "float32")
    Y = T.match_sparse_buffer(y, (I, F), "float32")
    with T.iter([I, F, J, T.attach(I_T, J)], "SSRR", "hyper_gnn") as [vi, vf, vj, vi_t]:
        with T.init():
            Y[vi, vf] = T.float32(0)
        Y[vi, vf] = Y[vi, vf] + X[vi_t, vf]


# @T.prim_func
# def bmm(
#     x: T.handle,
#     y: T.handle,
#     z: T.handle,
#     indptr_i: T.handle,
#     indptr_j: T.handle,
#     indptr_k: T.handle,
#     offset_ij: T.handle,
#     offset_jk: T.handle,
#     offset_ik: T.handle,
#     batch_size: T.int32,
#     nnz_i: T.int32,
#     nnz_j: T.int32,
#     nnz_k: T.int32,
#     nnz_ij: T.int32,
#     nnz_jk: T.int32,
#     nnz_ik: T.int32,
# ) -> None:
#     T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
#     B = T.dense_fixed(batch_size)
#     I = T.dense_variable(B, (32768, nnz_i), indptr_i, "int32")
#     J = T.dense_variable(B, (32768, nnz_j), indptr_j, "int32")
#     K = T.dense_variable(B, (32768, nnz_k), indptr_k, "int32")
#     (IJ,) = T.flatten([I, J], nnz_ij, (offset_ij,))
#     (JK,) = T.flatten([J, K], nnz_jk, (offset_jk,))
#     (IK,) = T.flatten([I, K], nnz_ik, (offset_ik,))
#     X = T.match_sparse_buffer(x, (B, IJ, J), "float32")
#     Y = T.match_sparse_buffer(y, (B, JK, K), "float32")
#     Z = T.match_sparse_buffer(z, (B, IK, K), "float32")
#     with T.iter([B, I, J, K], "SSRS", "bmm") as [vb, vi, vj, vk]:
#         with T.init():
#             Z[vb, vi, vk] = 0.0
#         Z[vb, vi, vk] = Z[vb, vi, vk] + X[vb, vi, vj] * Y[vb, vj, vk]


@T.prim_func
def sddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, K), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K), "float32")
    C = T.match_sparse_buffer(c, (I, J), "float32")

    with T.iter([I, J, K], "SSR", "sddmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def fused_sddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, K), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K), "float32")
    C = T.match_sparse_buffer(c, (I, J), "float32")

    with T.iter([T.fuse(I, J), K], "SSR", "sddmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def square_sum(
    a: T.handle,
    b: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    indptr_k: T.handle,
    indices_k: T.handle,
    nnz_j: T.int32,
    nnz_k: T.int32,
    M: T.int32,
    N1: T.int32,
    N2: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(M)
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), "int32")
    K = T.sparse_variable(J, (N2, nnz_k), (indptr_k, indices_k), "int32")
    A = T.match_sparse_buffer(a, (I, J, K), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")

    with T.iter([I, J, K], "SRR", "square_sum") as [vi, vj, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj, vk]


@T.prim_func
def square_sum_two_K(
    a: T.handle,
    b: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    indptr_k0: T.handle,
    indices_k0: T.handle,
    indptr_k1: T.handle,
    indices_k1: T.handle,
    nnz_j: T.int32,
    nnz_k: T.int32,
    M: T.int32,
    N1: T.int32,
    N2: T.int32,
):
    # Used only for testing `GetIndicesRange()`.
    # Currently it is ensured that `indptr_k0` is the same as `indptr_k1`, and `indices_k0` is the
    # same as `indices_k1`.
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(M)
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), "int32")
    K0 = T.sparse_variable(J, (N2, nnz_k), (indptr_k0, indices_k0), "int32")
    K1 = T.sparse_variable(J, (N2, nnz_k), (indptr_k1, indices_k1), "int32")
    A = T.match_sparse_buffer(a, (I, J, K0), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")

    with T.iter([I, J, K1], "SRR", "square_sum") as [vi, vj, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj, vk]


@T.prim_func
def fused_reduction_4d_2d(
    x: T.handle,
    y: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_l: T.handle,
    n: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_l: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    J = T.dense_variable(I, (32768, nnz_j), indptr_j, "int32")
    K = T.dense_variable(J, (32768, nnz_k), indptr_k, "int32")
    L = T.dense_variable(K, (32768, nnz_l), indptr_l, "int32")
    X = T.match_sparse_buffer(x, (I, J, K, L), "float32")
    Y = T.match_sparse_buffer(y, (I, J), "float32")
    with T.iter([T.fuse(I, J), K, L], "SSRR", "reduction_4d_2d") as [vi, vj, vk, vl]:
        with T.init():
            Y[vi, vj] = 0.0
        Y[vi, vj] = Y[vi, vj] + X[vi, vj, vk, vl]


@T.prim_func
def fused_reduction_4d_3d(
    x: T.handle,
    y: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_l: T.handle,
    n: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_l: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    J = T.dense_variable(I, (32768, nnz_j), indptr_j, "int32")
    K = T.dense_variable(J, (32768, nnz_k), indptr_k, "int32")
    L = T.dense_variable(K, (32768, nnz_l), indptr_l, "int32")
    X = T.match_sparse_buffer(x, (I, J, K, L), "float32")
    Y = T.match_sparse_buffer(y, (I, J, K), "float32")
    with T.iter([T.fuse(I, J, K), L], "SSSR", "reduction_4d_3d") as [vi, vj, vk, vl]:
        with T.init():
            Y[vi, vj, vk] = 0.0
        Y[vi, vj, vk] = Y[vi, vj, vk] + X[vi, vj, vk, vl]


@T.prim_func
def rgcn_forward(
    etype: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    r: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    R = T.dense_fixed(r)
    F_in = T.dense_fixed(feat_size)
    F_out = T.dense_fixed(feat_size)
    E = T.match_sparse_buffer(etype, (I, J), "int32")
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float32")
    Y = T.match_sparse_buffer(y, (I, F_out), "float32")
    with T.iter([I, F_out, J, F_in], "SSRR", "rgcn-forward") as [
        vi,
        vout,
        vj,
        vin,
    ]:
        with T.init():
            Y[vi, vout] = 0.0
        Y[vi, vout] = Y[vi, vout] + W[E[vi, vj], vout, vin] * X[vj, vin]


@T.prim_func
def rgcn_hetero_forward(
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    n: T.int32,
    r: T.int32,
    feat_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    R = T.dense_fixed(r)
    I = T.sparse_variable(R, (n, nnz_i), (indptr_i, indices_i), "int32")
    J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
    I_detach = T.dense_fixed(n)
    J_detach = T.dense_fixed(n)
    F_in = T.dense_fixed(feat_size)
    F_out = T.dense_fixed(feat_size)
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float32")
    Y = T.match_sparse_buffer(y, (I_detach, F_out), "float32")
    with T.iter([F_out, R, I, J, F_in], "SSSRR", "rgcn-hetero-forward") as [vout, vr, vi, vj, vin]:
        with T.init():
            Y[vi, vout] = 0.0
        Y[vi, vout] = Y[vi, vout] + W[vr, vout, vin] * X[vj, vin]


@T.prim_func
def rgcn_hetero_forward_2(
    w: T.handle,
    x: T.handle,
    y: T.handle,
    etypes: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    n: T.int32,
    r: T.int32,
    group: T.int32,
    feat_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    R = T.dense_fixed(r)
    G = T.dense_fixed(group)
    I = T.sparse_variable(G, (n, nnz_i), (indptr_i, indices_i), "int32")
    J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
    I_detach = T.dense_fixed(n)
    J_detach = T.dense_fixed(n)
    F_in = T.dense_fixed(feat_size)
    F_out = T.dense_fixed(feat_size)
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float32")
    Y = T.match_sparse_buffer(y, (I_detach, F_out), "float32")
    E = T.match_sparse_buffer(etypes, (G,), "int32")
    with T.iter([F_out, G, I, J, F_in], "SSSRR", "rgcn-hetero-forward") as [vout, vg, vi, vj, vin]:
        with T.init():
            Y[vi, vout] = 0.
        Y[vi, vout] = Y[vi, vout] + W[E[vg], vout, vin] * X[vj, vin]


@T.prim_func
def sparse_softmax(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    nnz: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I, J), "float32")
    TMP = T.alloc_sparse_buffer((I,), "float32", "global")
    TMP1 = T.alloc_sparse_buffer((I,), "float32", "global")
    with T.iter([I], "S", "sparse_softmax") as [vi]:
        with T.iter([J], "R", "computer_max") as [vj]:
            with T.init():
                TMP[vi] = T.float32(-100000)
            TMP[vi] = T.max(TMP[vi], A[vi, vj])
        with T.iter([J], "R", "exp_and_sum") as [vj]:
            with T.init():
                TMP1[vi] = T.float32(-100000)
            TMP1[vi] = TMP1[vi] + T.exp(A[vi, vj] - TMP[vi], dtype="float32")
        with T.iter([J], "S", "div") as [vj]:
            B[vi, vj] = T.exp(A[vi, vj], dtype="float32") / TMP1[vi]


@T.prim_func
def csr2bsr(
    a: T.handle,
    b: T.handle,
    indptr_in: T.handle,
    indices_in: T.handle,
    indptr_out: T.handle,
    indices_out: T.handle,
    m_in: T.int32,
    n_in: T.int32,
    m_out: T.int32,
    n_out: T.int32,
    nnz_in: T.int32,
    nnz_out: T.int32,
    blk_size: T.int32
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m_in)
    J = T.sparse_variable(I, (n_in, nnz_in), (indptr_in, indices_in), "int32")
    I_bsr = T.dense_fixed(m_out)
    J_bsr = T.sparse_variable(I_bsr, (n_out, nnz_out), (indptr_out, indices_out), "int32")
    BI = T.dense_fixed(blk_size)
    BJ = T.dense_fixed(blk_size)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I_bsr, J_bsr, BI, BJ), "float32")
    with T.iter([I, J], "SS", "csr2bsr") as [vi, vj]:
        B[vi // blk_size, vj // blk_size, vi % blk_size, vj % blk_size] = A[vi, vj]
 