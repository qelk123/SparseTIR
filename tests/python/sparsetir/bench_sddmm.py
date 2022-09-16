from statistics import mean
import dgl
import tvm
import argparse
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import torch as th
from tvm.script import tir as T
import tvm.sparse
from ogb.nodeproppred import DglNodePropPredDataset
from sparse_tir_lowered_iter_scripts import fused_sddmm
from torch.profiler import profile, ProfilerActivity, schedule


class TorchOpTimer(object):
    def __enter__(self):
        th.cuda.synchronize()
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event) / 1e3


def bench_sddmm(g: dgl.DGLGraph, feat_size: int):
    indptr, indices, _ = g.adj_sparse("csr")
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    nnz = g.number_of_edges()

    M, N, F, NNZ = fused_sddmm.params[-4:]
    a = th.rand(m, feat_size).to(th.float32)
    b = th.rand(n, feat_size).to(th.float32)
    c = th.zeros(nnz).to(th.float32)

    # dgl
    a_gpu = a.to(0)
    b_gpu = b.to(0)
    g = g.to(0)

    with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=0, warmup=10, active=100)) as prof:
        with th.no_grad():
            for epoch in range(100):
                c_golden = dgl.ops.u_dot_v(g, a_gpu, b_gpu)
                prof.step()
    print(prof.key_averages())
    return

    # accum_time = 0.0
    # runs = 0
    # cold_start_time = 3

    # for i in range(10):
    #     with TorchOpTimer() as timer:
    #         c_golden = dgl.ops.u_dot_v(g, a_gpu, b_gpu)
    #     if i >= cold_start_time:
    #         accum_time += timer.time
    #         runs += 1
    # print("dgl:\t\t", accum_time / runs * 1000)

    # tvm
    mod = tvm.IRModule.from_expr(fused_sddmm.specialize({M: m, N: n, F: feat_size, NNZ: nnz}))

    # split preprocess and compute
    mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
    mod_sddmm = tvm.tir.transform.RemovePreprocess()(mod)

    # schedule preprocess
    sch = tir.Schedule(mod_preprocess)
    blk = sch.get_block("binary_search_block_0_0")
    (i,) = sch.get_loops(blk)
    io, ii = sch.split(i, [None, 32])
    sch.bind(ii, "threadIdx.x")
    sch.bind(io, "blockIdx.x")
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    preproc = tvm.build(mod["main"], target="cuda")

    # compute mid
    a_nd = tvm.nd.array(a.view(-1).numpy(), tvm.cuda())
    b_nd = tvm.nd.array(b.view(-1).numpy(), tvm.cuda())
    c_nd = tvm.nd.array(c.numpy(), tvm.cuda())
    indptr_nd = tvm.nd.array(indptr.numpy(), tvm.cuda())
    indices_nd = tvm.nd.array(indices.numpy(), tvm.cuda())
    mid_nd = tvm.nd.array(np.zeros((nnz,), np.int32), tvm.cuda())

    preproc(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)

    best = 1e9
    for ty in [1, 2, 4, 8]:
        for tx in [8, 16, 32]:
            for vec_size in [1, 2, 4]:
                for group_size in [1, 2, 4]:
                    if tx * vec_size > feat_size:
                        continue
                    # schedule compute
                    sch = tir.Schedule(mod_sddmm)
                    blk = sch.get_block("sddmm0")
                    j, k = sch.get_loops(blk)
                    ko, kio, kii = sch.split(k, [None, tx, vec_size])
                    rf_blk = sch.rfactor(kio, 2)
                    j = sch.get_loops(rf_blk)[0]
                    joo, joi, ji = sch.split(j, [None, ty, group_size])
                    sch.bind(joo, "blockIdx.x")
                    sch.bind(joi, "threadIdx.y")
                    sch.unroll(ji)
                    sch.reverse_compute_at(blk, joi, True)
                    sch.set_scope(rf_blk, 0, "local")
                    read_A = sch.cache_read(rf_blk, 0, "local")
                    read_B = sch.cache_read(rf_blk, 2, "local")
                    write_C = sch.cache_write(blk, 0, "local")
                    ko, kio, kii = sch.get_loops(rf_blk)[-3:]
                    sch.reorder(ko, ji)
                    # schedule read A
                    sch.compute_at(read_A, ji, True)
                    ax0, ax1 = sch.split(sch.get_loops(read_A)[-1], [tx, vec_size])
                    sch.bind(ax0, "threadIdx.x")
                    sch.vectorize(ax1)
                    # schedule read B
                    sch.compute_at(read_B, ji, True)
                    ax0, ax1 = sch.split(sch.get_loops(read_B)[-1], [tx, vec_size])
                    sch.bind(ax0, "threadIdx.x")
                    sch.vectorize(ax1)
                    # schedule write C
                    sch.reverse_compute_at(write_C, joi, True)
                    ax0, ax1 = sch.get_loops(write_C)[-2:]
                    sch.vectorize(ax1)
                    # schedule rf
                    sch.bind(kio, "threadIdx.x")
                    sch.unroll(kii)
                    sch.unroll(ko)
                    # schedule write back
                    ax0, ax1, ax2 = sch.get_loops(blk)[-3:]
                    sch.reorder(ax1, ax2, ax0)
                    sch.bind(ax0, "threadIdx.x")
                    sch.unroll(ax2)
                    sch.unroll(ax1)
                    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
                    sddmm = tvm.build(mod["main"], target="cuda")

                    # check result
                    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd]
                    sddmm(*args)
                    tvm.testing.assert_allclose(c_nd.numpy(), c_golden.view(-1).cpu(), rtol=1e-5)

                    # evaluate time
                    evaluator = sddmm.time_evaluator(sddmm.entry_name, tvm.cuda(0), number=10)
                    mean_time = evaluator(*args).mean * 1000

                    if mean_time < best:
                        best = mean_time
                        best_config = (tx, ty, vec_size, group_size)
    print("sparse tir:\t{}".format(best))
    print("best config:\t{}".format(best_config))


def get_dataset(name: str):
    if name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
        g = arxiv[0][0]
    elif name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins")
        g = proteins[0][0]
    elif name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0]
    elif name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi)
    elif name == "reddit":
        reddit = dgl.data.RedditDataset()
        g = reddit[0]
    else:
        raise KeyError("Unknown dataset {}.".format(name))
    g = dgl.graph(g.edges("uv", "srcdst"), num_nodes=g.num_nodes())
    return g.int()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sddmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default='pubmed', help="dataset name")
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)
    for feat_size in [32, 64, 128, 256, 512]:
        bench_sddmm(g, feat_size)