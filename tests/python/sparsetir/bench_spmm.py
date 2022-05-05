from dgl._deprecate.graph import DGLGraph
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import torch as th
import dgl
from tvm.script import tir as T
import tvm.sparse
from ogb.nodeproppred import DglNodePropPredDataset


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event) / 1e3


def pad_graph(g: DGLGraph, tile_size=32) -> DGLGraph:
    u, v = g.edges()
    rows = [u.flatten()]
    cols = [v.flatten()]

    for node_id, deg in enumerate(g.in_degrees().tolist()):
        edges_to_add = ((deg + tile_size - 1) // tile_size) * tile_size - deg
        rows.append(th.full((edges_to_add,), 0))
        cols.append(th.full((edges_to_add,), node_id))

    rows = th.cat(rows)
    cols = th.cat(cols)

    return dgl.graph((rows, cols), num_nodes=g.num_dst_nodes())


@T.prim_func
def csrmm_tir(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    M: T.int32,
    N: T.int32,
    K: T.int32,
    NNZ: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZ,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZ,), "int32")
    for i, k in T.grid(M, K):
        with T.block("spmm_outer"):
            vi, vk = T.axis.remap("SS", [i, k])
            with T.init():
                C[vi * K + vk] = 0.0
            for j in T.serial(0, A_indptr[vi + 1] - A_indptr[vi]):
                with T.block("spmm_inner"):
                    T.block_attr({"sparse": True})
                    vj = T.axis.R(NNZ, j + A_indptr[vi])
                    C[vi * K + vk] = C[vi * K + vk] + A_data[vj] * B[A_indices[vj] * K + vk]


@T.prim_func
def csrmm_padding_tir(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    M: T.int32,
    N: T.int32,
    K: T.int32,
    NNZT: T.int32,
    tile_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZT * tile_size,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZT * tile_size,), "int32")
    for i, k in T.grid(M, K):
        with T.block("spmm_outer"):
            vi, vk = T.axis.remap("SS", [i, k])
            with T.init():
                C[vi * K + vk] = 0.0
            for j in T.grid(A_indptr[vi + 1] - A_indptr[vi]):
                with T.block("spmm_inner"):
                    T.block_attr({"sparse": True})
                    vj = T.axis.remap("R", [j])
                    for t in T.grid(tile_size):
                        with T.block("spmm_inner_2"):
                            vt = T.axis.remap("R", [t])
                            C[vi * K + vk] = (
                                C[vi * K + vk]
                                + A_data[(vj + A_indptr[vi]) * tile_size + vt]
                                * B[A_indices[(vj + A_indptr[vi]) * tile_size + vt] * K + vk]
                            )


@T.prim_func
def csrmm_hyb_tir(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    placeholder: T.handle,
    row_32: T.handle,
    indices_32: T.handle,
    a_32: T.handle,
    row_64: T.handle,
    indices_64: T.handle,
    a_64: T.handle,
    row_128: T.handle,
    indices_128: T.handle,
    a_128: T.handle,
    row_256: T.handle,
    indices_256: T.handle,
    a_256: T.handle,
    row_512: T.handle,
    indices_512: T.handle,
    a_512: T.handle,
    n32: T.int32,
    n64: T.int32,
    n128: T.int32,
    n256: T.int32,
    n512: T.int32,
    n: T.int32,
    f: T.int32,
):
    T.func_attr(
        {
            "global_symbol": "main",
            "tir.noalias": True,
            "sparse_tir_level": 2,
            "horizontal_fuse": True,
        }
    )
    N = T.dense_fixed(n)
    O = T.dense_fixed(1)
    F = T.dense_fixed(f)
    I32 = T.sparse_variable(O, (n, n32), (placeholder, row_32))
    J32 = T.sparse_fixed(I32, (n, 32), indices_32)
    A32 = T.match_sparse_buffer(a_32, [O, I32, J32], "float32")
    I64 = T.sparse_variable(O, (n, n64), (placeholder, row_64))
    J64 = T.sparse_fixed(I64, (n, 64), indices_64)
    A64 = T.match_sparse_buffer(a_64, [O, I64, J64], "float32")
    I128 = T.sparse_variable(O, (n, n128), (placeholder, row_128))
    J128 = T.sparse_fixed(I128, (n, 128), indices_128)
    A128 = T.match_sparse_buffer(a_128, [O, I128, J128], "float32")
    I256 = T.sparse_variable(O, (n, n256), (placeholder, row_256))
    J256 = T.sparse_fixed(I256, (n, 256), indices_256)
    A256 = T.match_sparse_buffer(a_256, [O, I256, J256], "float32")
    I512 = T.sparse_variable(O, (n, n512), (placeholder, row_512))
    J512 = T.sparse_fixed(I512, (n, 512), indices_512)
    A512 = T.match_sparse_buffer(a_512, [O, I512, J512], "float32")
    B = T.match_sparse_buffer(b, (N, F), "float32")
    C = T.match_sparse_buffer(c, (N, F), "float32")
    with T.iter([T.fuse(O, I32), J32, F], "SSRS", "csrmm_32") as [vo, vi, vj, vf]:
        with T.init():
            C[vi, vf] = 0.0
        C[vi, vf] = C[vi, vf] + A32[vo, vi, vj] * B[vj, vf]
    with T.iter([T.fuse(O, I64), J64, F], "SSRS", "csrmm_64") as [vo, vi, vj, vf]:
        with T.init():
            C[vi, vf] = 0.0
        C[vi, vf] = C[vi, vf] + A64[vo, vi, vj] * B[vj, vf]
    with T.iter([T.fuse(O, I128), J128, F], "SSRS", "csrmm_128") as [vo, vi, vj, vf]:
        with T.init():
            C[vi, vf] = 0.0
        C[vi, vf] = C[vi, vf] + A128[vo, vi, vj] * B[vj, vf]
    with T.iter([T.fuse(O, I256), J256, F], "SSRS", "csrmm_256") as [vo, vi, vj, vf]:
        with T.init():
            C[vi, vf] = 0.0
        C[vi, vf] = C[vi, vf] + A256[vo, vi, vj] * B[vj, vf]
    with T.iter([T.fuse(O, I512), J512, F], "SSRS", "csrmm_512") as [vo, vi, vj, vf]:
        with T.init():
            C[vi, vf] = 0.0
        C[vi, vf] = C[vi, vf] + A512[vo, vi, vj] * B[vj, vf]


def bench_hyb(g, feat_size=128):
    # still work in progress
    in_degrees = g.in_degrees()
    rows_32 = (in_degrees <= 32).nonzero().view(-1)
    rows_64 = (in_degrees <= 64).nonzero().view(-1)
    rows_128 = (in_degrees <= 128).nonzero().view(-1)
    rows_256 = (in_degrees <= 256).nonzero().view(-1)
    rows_512 = (in_degrees > 256).nonzero().view(-1)
    n_32 = len(rows_32)
    n_64 = len(rows_64)
    n_128 = len(rows_128)
    n_256 = len(rows_256)
    n_512 = len(rows_512)

    N32, N64, N128, N256, N512, N, F = csrmm_hyb_tir.params[-7:]

    mod = tvm.IRModule.from_expr(
        csrmm_hyb_tir.specialize(
            {
                N32: n_32,
                N64: n_64,
                N128: n_128,
                N256: n_256,
                N512: n_512,
                N: g.num_nodes(),
                F: feat_size,
            }
        )
    )
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    # schedule 32
    blk_32 = sch.get_block("csrmm_320")
    i, j, f = sch.get_loops(blk_32)
    sch.reorder(f, j)
    sch.bind(f, "threadIdx.x")
    # sch.unroll(j)
    sch.bind(i, "blockIdx.x")
    # sch.cache_write(blk_32, 0, "local")
    # schedule 64
    blk_64 = sch.get_block("csrmm_640")
    i, j, f = sch.get_loops(blk_64)
    sch.reorder(f, j)
    sch.bind(f, "threadIdx.x")
    # sch.unroll(j)
    sch.bind(i, "blockIdx.x")
    # schedule 128
    blk_128 = sch.get_block("csrmm_1280")
    i, j, f = sch.get_loops(blk_128)
    sch.reorder(f, j)
    sch.bind(f, "threadIdx.x")
    # sch.unroll(j)
    sch.bind(i, "blockIdx.x")
    # schedule 256
    blk_256 = sch.get_block("csrmm_2560")
    i, j, f = sch.get_loops(blk_256)
    sch.reorder(f, j)
    sch.bind(f, "threadIdx.x")
    # sch.unroll(j)
    sch.bind(i, "blockIdx.x")
    # schedule 512
    blk_512 = sch.get_block("csrmm_5120")
    i, j, f = sch.get_loops(blk_512)
    sch.reorder(f, j)
    sch.bind(f, "threadIdx.x")
    # sch.unroll(j)
    sch.bind(i, "blockIdx.x")

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    # print(mod["main"].script())
    f = tvm.build(mod, target="cuda")
    print(f.imported_modules[0].get_source())


def bench_tir_csrmm(g, feat_size=128):
    # generate random input
    indptr, indices, _ = g.adj_sparse("csc")

    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    k = feat_size
    nnz = g.num_edges()
    x = np.random.rand(n, k).astype("float32")
    y = np.zeros((m * k,)).astype("float32")

    # specialize function
    _, _, _, _, _, M, N, K, NNZ = csrmm_tir.params
    sch = tir.Schedule(csrmm_tir.specialize({M: m, N: n, K: k, NNZ: nnz}))
    blk_outer = sch.get_block("spmm_outer")
    i, k = sch.get_loops(blk_outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    A_indptr = tvm.nd.array(indptr.numpy().astype("int32"), device=tvm.cuda(0))
    A_indices = tvm.nd.array(indices.numpy().astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(np.ones((nnz,)).astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target="cuda")
    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    for i in range(10):
        with TorchOpTimer() as timer:
            f(A_data, X_nd, Y_nd, A_indptr, A_indices)
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1
    print("tir naive time: {:.3f}ms".format(accum_time / runs * 1000))

    g_gpu = g.to(0)
    h = th.from_numpy(x).to(0)
    weight = th.ones(nnz).to(0)
    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    for i in range(10):
        with TorchOpTimer() as timer:
            out = dgl.ops.u_mul_e_sum(g_gpu, h, weight)
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1
    print("cusparse time: {:.3f}ms".format(accum_time / runs * 1000))

    for tile_size in [8, 16, 24, 32, 40, 48, 56, 64]:
        g_pad = pad_graph(g, tile_size=tile_size)
        m = g_pad.num_src_nodes()
        n = g_pad.num_dst_nodes()
        k = feat_size
        x = np.random.rand(n, k).astype("float32")
        y = np.zeros((m * k,)).astype("float32")
        nnzt = g_pad.num_edges() // tile_size
        indptr_, indices_, _ = g_pad.adj_sparse("csc")
        _, _, _, _, _, M, N, K, NNZT, TILE_SIZE = csrmm_padding_tir.params
        sch = tir.Schedule(
            csrmm_padding_tir.specialize({M: m, N: n, K: k, NNZT: nnzt, TILE_SIZE: tile_size})
        )
        # print(sch.mod["main"].script())
        blk_outer = sch.get_block("spmm_outer")
        i, k = sch.get_loops(blk_outer)
        koo, ko, ki = sch.split(k, [None, 2, 32])
        blk_inner_2 = sch.get_block("spmm_inner_2")
        (t,) = sch.get_loops(blk_inner_2)
        sch.unroll(t)
        sch.bind(ko, "vthread.x")
        sch.bind(i, "blockIdx.x")
        sch.bind(koo, "blockIdx.y")
        sch.bind(ki, "threadIdx.x")

        # convert numpy tensor to tvm ndarray
        A_indptr = tvm.nd.array((indptr_.numpy() // tile_size).astype("int32"), device=tvm.cuda(0))
        A_indices = tvm.nd.array(indices_.numpy().astype("int32"), device=tvm.cuda(0))
        A_data = tvm.nd.array(np.ones((nnzt * tile_size,)).astype("float32"), device=tvm.cuda(0))
        X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
        Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

        # build function
        f = tvm.build(sch.mod, target="cuda")
        # print(f.imported_modules[0].get_source())
        accum_time = 0.0
        runs = 0
        cold_start_time = 3
        for i in range(10):
            with TorchOpTimer() as timer:
                f(A_data, X_nd, Y_nd, A_indptr, A_indices)
            if i >= cold_start_time:
                accum_time += timer.time
                runs += 1
        print(
            "tir w/ padding (tile_size={}) time: {:.3f}ms".format(
                tile_size, accum_time / runs * 1000
            )
        )


if __name__ == "__main__":
    arxiv = DglNodePropPredDataset(name="ogbn-proteins")
    g = arxiv[0][0]
    in_degrees = g.in_degrees()
    # proteins = DglNodePropPredDataset(name='ogbn-proteins')
    # g = proteins[0][0]
    # pubmed = dgl.data.PubmedGraphDataset()
    # g = pubmed[0]
    # ppi = dgl.data.PPIDataset()
    # g = dgl.batch(ppi)
    # reddit = dgl.data.RedditDataset()
    # g = reddit[0]

    # bench_hyb(g, feat_size=128)
    for feat_size in [32, 64, 128, 256, 512]:
        print('feat_size=', feat_size)
        bench_tir_csrmm(g, feat_size=feat_size)
