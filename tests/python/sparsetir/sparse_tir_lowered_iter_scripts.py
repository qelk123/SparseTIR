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
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    J_detach = T.dense_fixed(n, idtype="int32")
    K = T.dense_fixed(feat_size, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, K], dtype="float32")
    C = T.match_sparse_buffer(c, [I, K], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    for i in T.serial(m):
        with T.block("csrmm0"):
            vi = T.axis.spatial(m, i)
            T.reads(J_indptr[vi : vi + 2], A[vi, 0:n], B[0:n, 0:feat_size], J_indices[vi, 0:n])
            T.writes(C[vi, 0:feat_size])
            T.block_attr({"sparse": True})
            for j, k in T.grid(J_indptr[vi + 1] - J_indptr[vi], feat_size):
                with T.block("csrmm1"):
                    vj = T.axis.reduce(n, j)
                    vk = T.axis.spatial(feat_size, k)
                    T.reads(A[vi, vj], B[J_indices[vi, vj], vk], J_indices[vi, vj])
                    T.writes(C[vi, vk])
                    T.block_attr({"sparse": True})
                    with T.init():
                        C[vi, vk] = T.float32(0)
                    C[vi, vk] = C[vi, vk] + A[vi, vj] * B[J_indices[vi, vj], vk]


@T.prim_func
def csrmm_dense_iter(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    J_detach = T.dense_fixed(n, idtype="int32")
    K = T.dense_fixed(feat_size, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, K], dtype="float32")
    C = T.match_sparse_buffer(c, [I, K], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    mid_0 = T.alloc_sparse_buffer([I, J_detach], dtype="int32", extra_storage=0)
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    T.assume_buffer_domain(mid_0, [0, n])
    for i, j in T.grid(m, n):
        with T.block("binary_search_block_0_0"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(J_indices[vi, 0 : J_indptr[vi + 1] - J_indptr[vi]])
            T.writes(mid_0[vi, vj])
            T.block_attr({"preprocess": True, "sparse": True})
            low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            low[0] = 0
            high[0] = J_indptr[vi + 1] - J_indptr[vi]
            mid_0[vi, vj] = low[0] + (high[0] - low[0]) // 2
            while low[0] < high[0]:
                if J_indices[vi, mid_0[vi, vj]] < vj:
                    low[0] = mid_0[vi, vj] + 1
                else:
                    high[0] = mid_0[vi, vj]
                mid_0[vi, vj] = low[0] + (high[0] - low[0]) // 2
    for i, j, k in T.grid(m, n, feat_size):
        with T.block("csrmm0"):
            vi, vj, vk = T.axis.remap("SRS", [i, j, k])
            T.reads(A[vi, mid_0[vi, vj]], mid_0[vi, vj], B[vj, vk])
            T.writes(C[vi, vk])
            T.block_attr({"sparse": True})
            with T.init():
                C[vi, vk] = T.float32(0)
            C[vi, vk] = C[vi, vk] + A[vi, mid_0[vi, vj]] * B[vj, vk]


@T.prim_func
def segment_reduce(a: T.handle, b: T.handle, indptr: T.handle, n: T.int32, nnz: T.int32) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(n, idtype="int32")
    J = T.dense_variable(I, (100, nnz), indptr, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [I], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz])
    for i in T.serial(n):
        with T.block("segment_reduce0"):
            vi = T.axis.spatial(n, i)
            T.reads(J_indptr[vi : vi + 2], A[vi, 0:100])
            T.writes(B[vi])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("segment_reduce1"):
                    vj = T.axis.reduce(100, j)
                    T.reads(A[vi, vj])
                    T.writes(B[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B[vi] = T.float32(0)
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
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(n, idtype="int32")
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (m, nnz), indptr, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [I], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, m])
    for i in T.serial(n):
        with T.block("csr_reduce0"):
            vi = T.axis.spatial(n, i)
            T.reads(J_indptr[vi : vi + 2], A[vi, 0:m])
            T.writes(B[vi])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("csr_reduce1"):
                    vj = T.axis.reduce(m, j)
                    T.reads(A[vi, vj])
                    T.writes(B[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B[vi] = T.float32(0)
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
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(nb, idtype="int32")
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (mb, nnzb), indptr, idtype="int32")
    J_detach = T.dense_fixed(mb, idtype="int32")
    BI = T.dense_fixed(blk, idtype="int32")
    BJ = T.dense_fixed(blk, idtype="int32")
    F = T.dense_fixed(feat_size, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J, BI, BJ], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, BJ, F], dtype="float32")
    C = T.match_sparse_buffer(c, [I, BI, F], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnzb])
    T.assume_buffer_domain(J_indices, [0, mb])
    for i, bi, bj, f in T.grid(nb, blk, blk, feat_size):
        with T.block("bsrmm0"):
            vi, vbi, vbj, vf = T.axis.remap("SSRS", [i, bi, bj, f])
            T.reads(
                J_indptr[vi : vi + 2], A[vi, 0:mb, vbi, vbj], B[0:mb, vbj, vf], J_indices[vi, 0:mb]
            )
            T.writes(C[vi, vbi, vf])
            T.block_attr({"sparse": True})
            with T.init():
                C[vi, vbi, vf] = T.float32(0)
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("bsrmm1"):
                    vj = T.axis.reduce(mb, j)
                    T.reads(A[vi, vj, vbi, vbj], B[J_indices[vi, vj], vbj, vf], J_indices[vi, vj])
                    T.writes(C[vi, vbi, vf])
                    T.block_attr({"sparse": True})
                    C[vi, vbi, vf] = (
                        C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[J_indices[vi, vj], vbj, vf]
                    )


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
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(nb, idtype="int32")
    J = T.sparse_fixed(I, (mb, col), indices, idtype="int32", sorted=True)
    J_dense = T.dense_fixed(col, idtype="int32")
    J_detach = T.dense_fixed(mb, idtype="int32")
    F = T.dense_fixed(feat_size, idtype="int32")
    BI = T.dense_fixed(blk, idtype="int32")
    BJ = T.dense_fixed(blk, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J, BI, BJ], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, BJ, F], dtype="float32")
    C = T.match_sparse_buffer(c, [I, BI, F], dtype="float32")
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indices, [0, mb])
    for i, j, bi, bj, f in T.grid(nb, col, blk, blk, feat_size):
        with T.block("ellmm0"):
            vi, vj, vbi, vbj, vf = T.axis.remap("SRSRS", [i, j, bi, bj, f])
            T.reads(A[vi, vj, vbi, vbj], B[J_indices[vi, vj], vbj, vf], J_indices[vi, vj])
            T.writes(C[vi, vbi, vf])
            T.block_attr({"sparse": True})
            with T.init():
                C[vi, vbi, vf] = T.float32(0)
            C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[J_indices[vi, vj], vbj, vf]


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
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [I, J], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    for i in T.serial(m):
        with T.block("csr_element_wise0"):
            vi = T.axis.spatial(m, i)
            T.reads(J_indptr[vi : vi + 2], A[vi, 0:n])
            T.writes(B[vi, 0:n])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("csr_element_wise1"):
                    vj = T.axis.spatial(n, j)
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    T.block_attr({"sparse": True})
                    B[vi, vj] = A[vi, vj] * T.float32(2.5)


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
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(n, idtype="int32")
    F = T.dense_fixed(feat_size, idtype="int32")
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (m, nnz), indptr, idtype="int32")
    J_detach = T.dense_fixed(m, idtype="int32")
    I_T = T.sparse_variable(J_detach, (n, nnz), (indptr_T, indices_T), idtype="int32", sorted=True)
    I_T_dense = T.dense_variable(J_detach, (n, nnz), indptr_T, idtype="int32")
    X = T.match_sparse_buffer(x, [I, F], dtype="float32")
    Y = T.match_sparse_buffer(y, [I, F], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    I_T_indptr = T.match_sparse_buffer(indptr_T, [J_detach], dtype="int32", extra_storage=1)
    I_T_indices = T.match_sparse_buffer(indices_T, [J_detach, I_T_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, m])
    T.assume_buffer_domain(I_T_indptr, [0, nnz])
    T.assume_buffer_domain(I_T_indices, [0, n])
    for i, f in T.grid(n, feat_size):
        with T.block("hyper_gnn0"):
            vi, vf = T.axis.remap("SS", [i, f])
            T.reads(
                J_indptr[vi : vi + 2],
                I_T_indptr[0:m],
                J_indices[vi, 0:m],
                X[0:n, vf],
                I_T_indices[0:m, 0:n],
            )
            T.writes(Y[vi, vf])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("hyper_gnn1"):
                    vj = T.axis.reduce(m, j)
                    T.reads(
                        I_T_indptr[J_indices[vi, vj] : J_indices[vi, vj] + 2],
                        J_indices[vi, vj],
                        X[0:n, vf],
                        I_T_indices[vj, 0:n],
                    )
                    T.writes(Y[vi, vf])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y[vi, vf] = T.float32(0)
                    for i_t in T.serial(
                        I_T_indptr[J_indices[vi, vj] + 1] - I_T_indptr[J_indices[vi, vj]]
                    ):
                        with T.block("hyper_gnn2"):
                            vi_t = T.axis.reduce(n, i_t)
                            T.reads(X[I_T_indices[vj, vi_t], vf], I_T_indices[vj, vi_t])
                            T.writes(Y[vi, vf])
                            T.block_attr({"sparse": True})
                            Y[vi, vf] = Y[vi, vf] + X[I_T_indices[vj, vi_t], vf]


@T.prim_func
def bmm(
    x: T.handle,
    y: T.handle,
    z: T.handle,
    indptr_i: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_ij: T.handle,
    indptr_ik: T.handle,
    indptr_jk: T.handle,
    batch_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_ij: T.int32,
    nnz_jk: T.int32,
    nnz_ik: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    B = T.dense_fixed(batch_size, idtype="int32")
    I = T.dense_variable(B, (32768, nnz_i), indptr_i, idtype="int32")
    J = T.dense_variable(B, (32768, nnz_j), indptr_j, idtype="int32")
    K = T.dense_variable(B, (32768, nnz_k), indptr_k, idtype="int32")
    IJ = T.dense_variable(B, (32768, nnz_ij), indptr_ij, idtype="int32")
    JK = T.dense_variable(B, (32768, nnz_jk), indptr_jk, idtype="int32")
    IK = T.dense_variable(B, (32768, nnz_ik), indptr_ik, idtype="int32")
    X = T.match_sparse_buffer(x, [B, I, IJ], dtype="float32")
    Y = T.match_sparse_buffer(y, [B, J, JK], dtype="float32")
    Z = T.match_sparse_buffer(z, [B, I, IK], dtype="float32")
    I_indptr = T.match_sparse_buffer(indptr_i, [B], dtype="int32", extra_storage=1)
    J_indptr = T.match_sparse_buffer(indptr_j, [B], dtype="int32", extra_storage=1)
    K_indptr = T.match_sparse_buffer(indptr_k, [B], dtype="int32", extra_storage=1)
    IJ_indptr = T.match_sparse_buffer(indptr_ij, [B], dtype="int32", extra_storage=1)
    IK_indptr = T.match_sparse_buffer(indptr_ik, [B], dtype="int32", extra_storage=1)
    JK_indptr = T.match_sparse_buffer(indptr_jk, [B], dtype="int32", extra_storage=1)
    # body
    # with T.block("root")
    T.assume_buffer_domain(I_indptr, [0, nnz_i])
    T.assume_buffer_domain(J_indptr, [0, nnz_j])
    T.assume_buffer_domain(K_indptr, [0, nnz_k])
    T.assume_buffer_domain(IJ_indptr, [0, nnz_ij])
    T.assume_buffer_domain(JK_indptr, [0, nnz_jk])
    T.assume_buffer_domain(IK_indptr, [0, nnz_ik])
    for b in T.serial(batch_size):
        with T.block("bmm0"):
            vb = T.axis.spatial(batch_size, b)
            T.reads(
                I_indptr[vb : vb + 2],
                J_indptr[vb : vb + 2],
                K_indptr[vb : vb + 2],
                X[vb, 0:32768, 0:32768],
                Y[vb, 0:32768, 0:32768],
            )
            T.writes(Z[vb, 0:32768, 0:32768])
            T.block_attr({"sparse": True})
            for i, j, k in T.grid(
                I_indptr[vb + 1] - I_indptr[vb],
                J_indptr[vb + 1] - J_indptr[vb],
                K_indptr[vb + 1] - K_indptr[vb],
            ):
                with T.block("bmm1"):
                    vi = T.axis.spatial(32768, i)
                    vj = T.axis.reduce(32768, j)
                    vk = T.axis.spatial(32768, k)
                    T.reads(X[vb, vi, vj], Y[vb, vj, vk])
                    T.writes(Z[vb, vi, vk])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Z[vb, vi, vk] = T.float32(0)
                    Z[vb, vi, vk] = Z[vb, vi, vk] + X[vb, vi, vj] * Y[vb, vj, vk]


@T.prim_func
def sddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    J_detach = T.dense_fixed(n, idtype="int32")
    K = T.dense_fixed(feat_size, idtype="int32")
    A = T.match_sparse_buffer(a, [I, K], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, K], dtype="float32")
    C = T.match_sparse_buffer(c, [I, J], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    for i in T.serial(m):
        with T.block("sddmm0"):
            vi = T.axis.spatial(m, i)
            T.reads(
                J_indptr[vi : vi + 2], A[vi, 0:feat_size], B[0:n, 0:feat_size], J_indices[vi, 0:n]
            )
            T.writes(C[vi, 0:n])
            T.block_attr({"sparse": True})
            for j, k in T.grid(J_indptr[vi + 1] - J_indptr[vi], feat_size):
                with T.block("sddmm1"):
                    vj = T.axis.spatial(n, j)
                    vk = T.axis.reduce(feat_size, k)
                    T.reads(A[vi, vk], B[J_indices[vi, vj], vk], J_indices[vi, vj])
                    T.writes(C[vi, vj])
                    T.block_attr({"sparse": True})
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[J_indices[vi, vj], vk]


@T.prim_func
def fused_sddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    J_detach = T.dense_fixed(n, idtype="int32")
    K = T.dense_fixed(feat_size, idtype="int32")
    A = T.match_sparse_buffer(a, [I, K], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, K], dtype="float32")
    C = T.match_sparse_buffer(c, [I, J], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    mid_0 = T.alloc_sparse_buffer([I, J], dtype="int32", extra_storage=0)
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    T.assume_buffer_domain(mid_0, [0, m])
    for j in T.serial(nnz):
        with T.block("binary_search_block_0_0"):
            vi = T.axis.spatial(1, 0)
            vj = T.axis.spatial(nnz, j)
            T.reads(J_indptr[0 : m + 1])
            T.writes(mid_0[vi, vj])
            T.block_attr({"preprocess": True, "sparse": True})
            low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            low[0] = 0
            high[0] = m + 1
            mid_0[vi, vj] = low[0] + (high[0] - low[0]) // 2
            while low[0] < high[0]:
                if J_indptr[mid_0[vi, vj]] > vj:
                    high[0] = mid_0[vi, vj]
                else:
                    low[0] = mid_0[vi, vj] + 1
                mid_0[vi, vj] = low[0] + (high[0] - low[0]) // 2
            mid_0[vi, vj] = mid_0[vi, vj] - 1
    for j, k in T.grid(nnz, feat_size):
        with T.block("sddmm0"):
            vi = T.axis.spatial(1, 0)
            vj, vk = T.axis.remap("SR", [j, k])
            T.reads(
                A[mid_0[vi, vj], vk], mid_0[vi, vj], B[J_indices[vi, vj], vk], J_indices[vi, vj]
            )
            T.writes(C[vi, vj])
            T.block_attr({"sparse": True})
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[mid_0[vi, vj], vk] * B[J_indices[vi, vj], vk]


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
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(M, idtype="int32")
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (N1, nnz_j), indptr_j, idtype="int32")
    K = T.sparse_variable(J, (N2, nnz_k), (indptr_k, indices_k), idtype="int32", sorted=True)
    K_dense = T.dense_variable(J, (N2, nnz_k), indptr_k, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J, K], dtype="float32")
    B = T.match_sparse_buffer(b, [I], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr_j, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices_j, [I, J_dense], dtype="int32")
    K_indptr = T.match_sparse_buffer(indptr_k, [I, J_dense], dtype="int32", extra_storage=1)
    K_indices = T.match_sparse_buffer(indices_k, [I, J, K_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz_j])
    T.assume_buffer_domain(J_indices, [0, N1])
    T.assume_buffer_domain(K_indptr, [0, nnz_k])
    T.assume_buffer_domain(K_indices, [0, N2])
    for i in T.serial(M):
        with T.block("square_sum0"):
            vi = T.axis.spatial(M, i)
            T.reads(J_indptr[vi : vi + 2], K_indptr[vi, 0 : N1 + 1], A[vi, 0:N1, 0:N2])
            T.writes(B[vi])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("square_sum1"):
                    vj = T.axis.reduce(N1, j)
                    T.reads(K_indptr[vi, vj : vj + 2], A[vi, vj, 0:N2])
                    T.writes(B[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B[vi] = T.float32(0)
                    for k in T.serial(K_indptr[vi, vj + 1] - K_indptr[vi, vj]):
                        with T.block("square_sum2"):
                            vk = T.axis.reduce(N2, k)
                            T.reads(A[vi, vj, vk])
                            T.writes(B[vi])
                            T.block_attr({"sparse": True})
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
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(M, idtype="int32")
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (N1, nnz_j), indptr_j, idtype="int32")
    K0 = T.sparse_variable(J, (N2, nnz_k), (indptr_k0, indices_k0), idtype="int32", sorted=True)
    K0_dense = T.dense_variable(J, (N2, nnz_k), indptr_k0, idtype="int32")
    K1 = T.sparse_variable(J, (N2, nnz_k), (indptr_k1, indices_k1), idtype="int32", sorted=True)
    K1_dense = T.dense_variable(J, (N2, nnz_k), indptr_k1, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J, K0], dtype="float32")
    B = T.match_sparse_buffer(b, [I], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr_j, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices_j, [I, J_dense], dtype="int32")
    K0_indptr = T.match_sparse_buffer(indptr_k0, [I, J_dense], dtype="int32", extra_storage=1)
    K0_indices = T.match_sparse_buffer(indices_k0, [I, J, K0_dense], dtype="int32")
    K1_indptr = T.match_sparse_buffer(indptr_k1, [I, J_dense], dtype="int32", extra_storage=1)
    K1_indices = T.match_sparse_buffer(indices_k1, [I, J, K1_dense], dtype="int32")
    # body
    # with T.block("root")
    mid_0 = T.alloc_sparse_buffer([I, J, K1], dtype="int32", extra_storage=0)
    T.assume_buffer_domain(J_indptr, [0, nnz_j])
    T.assume_buffer_domain(J_indices, [0, N1])
    T.assume_buffer_domain(K0_indptr, [0, nnz_k])
    T.assume_buffer_domain(K0_indices, [0, N2])
    T.assume_buffer_domain(K1_indptr, [0, nnz_k])
    T.assume_buffer_domain(K1_indices, [0, N2])
    T.assume_buffer_domain(mid_0, [0, N2])
    for i in T.serial(M):
        with T.block("binary_search_block_0_0"):
            vi = T.axis.spatial(M, i)
            T.reads(J_indptr[vi : vi + 2], K1_indptr[vi, 0 : N1 + 1], K0_indices[vi, 0:N1, 0:N2])
            T.writes(mid_0[vi, 0:N1, 0:N2])
            T.block_attr({"preprocess": True, "sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("binary_search_block_0_1"):
                    vj = T.axis.spatial(N1, j)
                    T.reads(
                        K1_indptr[vi, vj : vj + 2],
                        K0_indices[vi, vj, 0 : K0_indptr[vi, vj + 1] - K0_indptr[vi, vj]],
                    )
                    T.writes(mid_0[vi, vj, 0:N2])
                    T.block_attr({"preprocess": True, "sparse": True})
                    for k in T.serial(K1_indptr[vi, vj + 1] - K1_indptr[vi, vj]):
                        with T.block("binary_search_block_0_2"):
                            vk = T.axis.spatial(N2, k)
                            T.reads(
                                K0_indices[vi, vj, 0 : K0_indptr[vi, vj + 1] - K0_indptr[vi, vj]]
                            )
                            T.writes(mid_0[vi, vj, vk])
                            T.block_attr({"preprocess": True, "sparse": True})
                            low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                            high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                            low[0] = 0
                            high[0] = K0_indptr[vi, vj + 1] - K0_indptr[vi, vj]
                            mid_0[vi, vj, vk] = low[0] + (high[0] - low[0]) // 2
                            while low[0] < high[0]:
                                if K0_indices[vi, vj, mid_0[vi, vj, vk]] < K1_indices[vi, vj, vk]:
                                    low[0] = mid_0[vi, vj, vk] + 1
                                else:
                                    high[0] = mid_0[vi, vj, vk]
                                mid_0[vi, vj, vk] = low[0] + (high[0] - low[0]) // 2
    for i in T.serial(M):
        with T.block("square_sum0"):
            vi = T.axis.spatial(M, i)
            T.reads(
                J_indptr[vi : vi + 2],
                K1_indptr[vi, 0 : N1 + 1],
                A[vi, 0:N1, 0:N2],
                mid_0[vi, 0:N1, 0:N2],
            )
            T.writes(B[vi])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("square_sum1"):
                    vj = T.axis.reduce(N1, j)
                    T.reads(K1_indptr[vi, vj : vj + 2], A[vi, vj, 0:N2], mid_0[vi, vj, 0:N2])
                    T.writes(B[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B[vi] = T.float32(0)
                    for k in T.serial(K1_indptr[vi, vj + 1] - K1_indptr[vi, vj]):
                        with T.block("square_sum2"):
                            vk = T.axis.reduce(N2, k)
                            T.reads(A[vi, vj, mid_0[vi, vj, vk]], mid_0[vi, vj, vk])
                            T.writes(B[vi])
                            T.block_attr({"sparse": True})
                            B[vi] = B[vi] + A[vi, vj, mid_0[vi, vj, vk]]


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
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(n, idtype="int32")
    J = T.dense_variable(I, (32768, nnz_j), indptr_j, idtype="int32")
    K = T.dense_variable(J, (32768, nnz_k), indptr_k, idtype="int32")
    L = T.dense_variable(K, (32768, nnz_l), indptr_l, idtype="int32")
    X = T.match_sparse_buffer(x, [I, J, K, L], dtype="float32")
    Y = T.match_sparse_buffer(y, [I, J], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr_j, [I], dtype="int32", extra_storage=1)
    K_indptr = T.match_sparse_buffer(indptr_k, [I, J], dtype="int32", extra_storage=1)
    L_indptr = T.match_sparse_buffer(indptr_l, [I, J, K], dtype="int32", extra_storage=1)
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz_j])
    T.assume_buffer_domain(K_indptr, [0, nnz_k])
    T.assume_buffer_domain(L_indptr, [0, nnz_l])
    for j in T.serial(nnz_j):
        with T.block("reduction_4d_2d0"):
            vi = T.axis.spatial(1, 0)
            vj = T.axis.spatial(nnz_j, j)
            T.reads(
                K_indptr[vi, vj : vj + 2], L_indptr[vi, vj, 0:32769], X[vi, vj, 0:32768, 0:32768]
            )
            T.writes(Y[vi, vj])
            T.block_attr({"sparse": True})
            for k in T.serial(K_indptr[vi, vj + 1] - K_indptr[vi, vj]):
                with T.block("reduction_4d_2d1"):
                    vk = T.axis.reduce(32768, k)
                    T.reads(L_indptr[vi, vj, vk : vk + 2], X[vi, vj, vk, 0:32768])
                    T.writes(Y[vi, vj])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y[vi, vj] = T.float32(0)
                    for l in T.serial(L_indptr[vi, vj, vk + 1] - L_indptr[vi, vj, vk]):
                        with T.block("reduction_4d_2d2"):
                            vl = T.axis.reduce(32768, l)
                            T.reads(X[vi, vj, vk, vl])
                            T.writes(Y[vi, vj])
                            T.block_attr({"sparse": True})
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
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(n, idtype="int32")
    J = T.dense_variable(I, (32768, nnz_j), indptr_j, idtype="int32")
    K = T.dense_variable(J, (32768, nnz_k), indptr_k, idtype="int32")
    L = T.dense_variable(K, (32768, nnz_l), indptr_l, idtype="int32")
    X = T.match_sparse_buffer(x, [I, J, K, L], dtype="float32")
    Y = T.match_sparse_buffer(y, [I, J, K], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr_j, [I], dtype="int32", extra_storage=1)
    K_indptr = T.match_sparse_buffer(indptr_k, [I, J], dtype="int32", extra_storage=1)
    L_indptr = T.match_sparse_buffer(indptr_l, [I, J, K], dtype="int32", extra_storage=1)
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz_j])
    T.assume_buffer_domain(K_indptr, [0, nnz_k])
    T.assume_buffer_domain(L_indptr, [0, nnz_l])
    for k in T.serial(nnz_k):
        with T.block("reduction_4d_3d0"):
            vi = T.axis.spatial(1, 0)
            vj = T.axis.spatial(1, 0)
            vk = T.axis.spatial(nnz_k, k)
            T.reads(L_indptr[vi, vj, vk : vk + 2], X[vi, vj, vk, 0:32768])
            T.writes(Y[vi, vj, vk])
            T.block_attr({"sparse": True})
            for l in T.serial(L_indptr[vi, vj, vk + 1] - L_indptr[vi, vj, vk]):
                with T.block("reduction_4d_3d1"):
                    vl = T.axis.reduce(32768, l)
                    T.reads(X[vi, vj, vk, vl])
                    T.writes(Y[vi, vj, vk])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y[vi, vj, vk] = T.float32(0)
                    Y[vi, vj, vk] = Y[vi, vj, vk] + X[vi, vj, vk, vl]


@T.prim_func
def rgcn_homo_forward(
    etype: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    r: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    J_detach = T.dense_fixed(n, idtype="int32")
    R = T.dense_fixed(r, idtype="int32")
    F_in = T.dense_fixed(feat_size, idtype="int32")
    F_out = T.dense_fixed(feat_size, idtype="int32")
    E = T.match_sparse_buffer(etype, [I, J], dtype="int32")
    W = T.match_sparse_buffer(w, [R, F_out, F_in], dtype="float32")
    X = T.match_sparse_buffer(x, [J_detach, F_in], dtype="float32")
    Y = T.match_sparse_buffer(y, [I, F_out], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    for i, fo in T.grid(m, feat_size):
        with T.block("rgcn-homo-forward0"):
            vi, vfo = T.axis.remap("SS", [i, fo])
            T.reads(
                J_indptr[vi : vi + 2],
                W[0:r, vfo, 0:feat_size],
                E[vi, 0:n],
                X[0:n, 0:feat_size],
                J_indices[vi, 0:n],
            )
            T.writes(Y[vi, vfo])
            T.block_attr({"sparse": True})
            for j, fi in T.grid(J_indptr[vi + 1] - J_indptr[vi], feat_size):
                with T.block("rgcn-homo-forward1"):
                    vj = T.axis.reduce(n, j)
                    vfi = T.axis.reduce(feat_size, fi)
                    T.reads(
                        W[E[vi, vj], vfo, vfi],
                        E[vi, vj],
                        X[J_indices[vi, vj], vfi],
                        J_indices[vi, vj],
                    )
                    T.writes(Y[vi, vfo])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y[vi, vfo] = T.float32(0)
                    Y[vi, vfo] = Y[vi, vfo] + W[E[vi, vj], vfo, vfi] * X[J_indices[vi, vj], vfi]


@T.prim_func
def rgcn_hetero_forward(
    a: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rels: T.int32,
    feat_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    R = T.dense_fixed(num_rels, idtype="int32")
    I = T.sparse_variable(R, (m, nnz_i), (indptr_i, indices_i), idtype="int32", sorted=True)
    I_dense = T.dense_variable(R, (m, nnz_i), indptr_i, idtype="int32")
    J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz_j), indptr_j, idtype="int32")
    I_detach = T.dense_fixed(m, idtype="int32")
    J_detach = T.dense_fixed(n, idtype="int32")
    F_in = T.dense_fixed(feat_size, idtype="int32")
    F_out = T.dense_fixed(feat_size, idtype="int32")
    A = T.match_sparse_buffer(a, [R, I, J], dtype="float32")
    W = T.match_sparse_buffer(w, [R, F_out, F_in], dtype="float32")
    X = T.match_sparse_buffer(x, [J_detach, F_in], dtype="float32")
    Y = T.match_sparse_buffer(y, [I_detach, F_out], dtype="float32")
    I_indptr = T.match_sparse_buffer(indptr_i, [R], dtype="int32", extra_storage=1)
    I_indices = T.match_sparse_buffer(indices_i, [R, I_dense], dtype="int32")
    J_indptr = T.match_sparse_buffer(indptr_j, [R, I_dense], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices_j, [R, I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(I_indptr, [0, nnz_i])
    T.assume_buffer_domain(I_indices, [0, m])
    T.assume_buffer_domain(J_indptr, [0, nnz_j])
    T.assume_buffer_domain(J_indices, [0, n])
    for fo, r in T.grid(feat_size, num_rels):
        with T.block("rgcn-hetero-forward0"):
            vfo, vr = T.axis.remap("SS", [fo, r])
            T.reads(
                I_indptr[vr : vr + 2],
                J_indptr[vr, 0 : m + 1],
                I_indices[vr, 0:m],
                A[vr, 0:m, 0:n],
                W[vr, vfo, 0:feat_size],
                X[0:n, 0:feat_size],
                J_indices[vr, 0:m, 0:n],
            )
            T.writes(Y[0:m, vfo])
            T.block_attr({"sparse": True})
            for i in T.serial(I_indptr[vr + 1] - I_indptr[vr]):
                with T.block("rgcn-hetero-forward1"):
                    vi = T.axis.spatial(m, i)
                    T.reads(
                        J_indptr[vr, vi : vi + 2],
                        I_indices[vr, vi],
                        A[vr, vi, 0:n],
                        W[vr, vfo, 0:feat_size],
                        X[0:n, 0:feat_size],
                        J_indices[vr, vi, 0:n],
                    )
                    T.writes(Y[I_indices[vr, vi], vfo])
                    T.block_attr({"sparse": True})
                    for j, fi in T.grid(J_indptr[vr, vi + 1] - J_indptr[vr, vi], feat_size):
                        with T.block("rgcn-hetero-forward2"):
                            vj = T.axis.reduce(n, j)
                            vfi = T.axis.reduce(feat_size, fi)
                            T.reads(
                                I_indices[vr, vi],
                                A[vr, vi, vj],
                                W[vr, vfo, vfi],
                                X[J_indices[vr, vi, vj], vfi],
                                J_indices[vr, vi, vj],
                            )
                            T.writes(Y[I_indices[vr, vi], vfo])
                            T.block_attr({"sparse": True})
                            with T.init():
                                Y[I_indices[vr, vi], vfo] = T.float32(0)
                            Y[I_indices[vr, vi], vfo] = (
                                Y[I_indices[vr, vi], vfo]
                                + A[vr, vi, vj] * W[vr, vfo, vfi] * X[J_indices[vr, vi, vj], vfi]
                            )


@T.prim_func
def sparse_softmax(
    a: T.handle, b: T.handle, indptr: T.handle, indices: T.handle, n: T.int32, nnz: T.int32
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(n, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [I, J], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    TMP = T.alloc_sparse_buffer([I], dtype="float32", extra_storage=0)
    TMP1 = T.alloc_sparse_buffer([I], dtype="float32", extra_storage=0)
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    for i in T.serial(n):
        with T.block("sparse_softmax0"):
            vi = T.axis.spatial(n, i)
            T.reads(J_indptr[vi : vi + 2], A[vi, 0:n], TMP[vi], TMP1[vi])
            T.writes(TMP[vi], TMP1[vi], B[vi, 0:n])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("computer_max0"):
                    vj = T.axis.reduce(n, j)
                    T.reads(A[vi, vj])
                    T.writes(TMP[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        TMP[vi] = T.min_value("float32")
                    TMP[vi] = T.max(TMP[vi], A[vi, vj])
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("exp_and_sum0"):
                    vj = T.axis.reduce(n, j)
                    T.reads(A[vi, vj], TMP[vi])
                    T.writes(TMP1[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        TMP1[vi] = T.float32(0)
                    TMP1[vi] = TMP1[vi] + T.exp(A[vi, vj] - TMP[vi], dtype="float32")
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("div0"):
                    vj = T.axis.spatial(n, j)
                    T.reads(A[vi, vj], TMP1[vi])
                    T.writes(B[vi, vj])
                    T.block_attr({"sparse": True})
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
    blk_size: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 1,
            "check_invalid_binary_search": False,
        }
    )
    I = T.dense_fixed(m_in, idtype="int32")
    J = T.sparse_variable(I, (n_in, nnz_in), (indptr_in, indices_in), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n_in, nnz_in), indptr_in, idtype="int32")
    I_bsr = T.dense_fixed(m_out, idtype="int32")
    J_bsr = T.sparse_variable(
        I_bsr, (n_out, nnz_out), (indptr_out, indices_out), idtype="int32", sorted=True
    )
    J_bsr_dense = T.dense_variable(I_bsr, (n_out, nnz_out), indptr_out, idtype="int32")
    BI = T.dense_fixed(blk_size, idtype="int32")
    BJ = T.dense_fixed(blk_size, idtype="int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [I_bsr, J_bsr, BI, BJ], dtype="float32")
    J_indptr = T.match_sparse_buffer(indptr_in, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices_in, [I, J_dense], dtype="int32")
    J_bsr_indptr = T.match_sparse_buffer(indptr_out, [I_bsr], dtype="int32", extra_storage=1)
    J_bsr_indices = T.match_sparse_buffer(indices_out, [I_bsr, J_bsr_dense], dtype="int32")
    # body
    # with T.block("root")
    mid_0 = T.alloc_sparse_buffer([I, J], dtype="int32", extra_storage=0)
    T.assume_buffer_domain(J_indptr, [0, nnz_in])
    T.assume_buffer_domain(J_indices, [0, n_in])
    T.assume_buffer_domain(J_bsr_indptr, [0, nnz_out])
    T.assume_buffer_domain(J_bsr_indices, [0, n_out])
    T.assume_buffer_domain(mid_0, [0, n_out])
    for i in T.serial(m_in):
        with T.block("binary_search_block_0_0"):
            vi = T.axis.spatial(m_in, i)
            T.reads(J_indptr[vi : vi + 2], J_bsr_indices[0:m_out, 0:n_out])
            T.writes(mid_0[vi, 0:n_in])
            T.block_attr({"preprocess": True, "sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("binary_search_block_0_1"):
                    vj = T.axis.spatial(n_in, j)
                    T.reads(
                        J_bsr_indices[
                            vi // blk_size,
                            0 : J_bsr_indptr[vi // blk_size + 1] - J_bsr_indptr[vi // blk_size],
                        ]
                    )
                    T.writes(mid_0[vi, vj])
                    T.block_attr({"preprocess": True, "sparse": True})
                    low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                    high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                    low[0] = 0
                    high[0] = J_bsr_indptr[vi // blk_size + 1] - J_bsr_indptr[vi // blk_size]
                    mid_0[vi, vj] = low[0] + (high[0] - low[0]) // 2
                    while low[0] < high[0]:
                        if (
                            J_bsr_indices[vi // blk_size, mid_0[vi, vj]]
                            < J_indices[vi, vj] // blk_size
                        ):
                            low[0] = mid_0[vi, vj] + 1
                        else:
                            high[0] = mid_0[vi, vj]
                        mid_0[vi, vj] = low[0] + (high[0] - low[0]) // 2
    for i in T.serial(m_in):
        with T.block("csr2bsr0"):
            vi = T.axis.spatial(m_in, i)
            T.reads(J_indptr[vi : vi + 2], A[vi, 0:n_in], mid_0[vi, 0:n_in], J_indices[vi, 0:n_in])
            T.writes(B[0:m_out, 0:n_out, 0:blk_size, 0:blk_size])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("csr2bsr1"):
                    vj = T.axis.spatial(n_in, j)
                    T.reads(A[vi, vj], mid_0[vi, vj], J_indices[vi, vj])
                    T.writes(B[0:m_out, mid_0[vi, vj], 0:blk_size, 0:blk_size])
                    T.block_attr({"sparse": True})
                    B[
                        vi // blk_size, mid_0[vi, vj], vi % blk_size, J_indices[vi, vj] % blk_size
                    ] = A[vi, vj]
