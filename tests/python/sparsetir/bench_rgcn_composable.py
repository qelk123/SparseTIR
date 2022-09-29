import enum
import dgl
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import dgl.function as fn
import torch as th
from tvm.script import tir as T
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from sparse_tir_scripts import rgcn_hetero_forward
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer, FormatRewriteRule
from typing import List, Tuple, Mapping
from sparse_tir_format_rewrite_scripts import ell3d


def get_dataset_by_name(name: str):
    if name == "aifb":
        return AIFBDataset()
    elif name == "mutag":
        return MUTAGDataset()
    elif name == "bgs":
        return BGSDataset()
    elif name == "am":
        return AMDataset()
    else:
        raise KeyError("Unknown dataset {}.".format(name))


def prepare_hetero_graph_simplified(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": th.IntTensor(ntype_pointer),
        "etype_edge_pointer": th.IntTensor(etype_pointer),
    }


def get_ground_truth(
    g: dgl.DGLHeteroGraph,
    type_pointers: Mapping[str, th.Tensor],
    feat: th.Tensor,
    weight: th.Tensor,
) -> th.Tensor:
    feat_size = feat.shape[-1]
    g_homo = dgl.to_homogeneous(g)
    g_homo = g_homo.to(0)
    weight_T = weight.permute(0, 2, 1).contiguous()
    etype_pointer = type_pointers["etype_edge_pointer"]
    try:
        g_homo.srcdata["feat"] = feat.unsqueeze(-1)
        us, vs = g_homo.edges()
        feat_transformed = feat[us]
        msg = th.zeros(g_homo.num_edges(), feat_size).to(0)
        with th.no_grad():
            for i in range(1, len(etype_pointer)):
                start = etype_pointer[i - 1]
                end = etype_pointer[i]
                msg[start:end] = feat_transformed[start:end] @ weight_T[i - 1]
            y_dgl_lowmem = dgl.ops.copy_e_sum(g_homo, msg)
    except RuntimeError as err:
        print("dgl-lowmem: OOM")
        y_dgl_lowmem = None
    return y_dgl_lowmem


def csf_to_ell3d_inv_idx_map(r, i, j):
    return r, i, j


def csf_to_ell3d_idx_map(r, i, j):
    return r, i, j


def test_rgcn_composable_format(
    g: dgl.DGLHeteroGraph,
    feat_size: int,
    feat: th.Tensor,
    weight: th.Tensor,
    ground_truth: th.Tensor,
):
    # preprocess data
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        indptr, indices, _ = g_sub.adj_sparse(fmt="csc")

        unique_nodes = th.nonzero(indptr[:-1] != indptr[1:]).squeeze(1)
        start = 0
        for end in range(0, len(unique_nodes)):
            pass

    # d0, d1, d2, nnz_1, nnz_2 = ell3d.params[-5:]
    nnz_cols_2 = ell3d.params[-1]
    rewrites = []
    for bucket_id, bucket_size in enumerate([1, 2, 4, 8]):
        rewrites.append(
            FormatRewriteRule(
                str(bucket_id),
                ell3d.specialize({
                    nnz_cols_2: bucket_size,
                }),
                ["A"],
                ["R", "I", "J"],
                ["R", "I", "J"],
                {"R": ["R"], "I": ["I"], "J": ["J"]},
                csf_to_ell3d_idx_map,
                csf_to_ell3d_inv_idx_map
            )
        )
    print(rewrites)
    mod = tvm.IRModule.from_expr(rgcn_hetero_forward)
    mod = tvm.tir.transform.SparseFormatRewrite(rewrites)(mod)
    print(mod["main"].script())
 


if __name__ == "__main__":
    feat_size = 32
    dataset = get_dataset_by_name("aifb")
    g = dataset[0]
    type_pointers = prepare_hetero_graph_simplified(g)
    n = g.num_nodes()
    r = len(g.etypes)
    feat = th.rand(n, feat_size).to(0) / 100
    weight = th.rand(r, feat_size, feat_size).to(0)
    # homograph
    ground_truth_y = get_ground_truth(g, type_pointers, feat, weight)
    test_rgcn_composable_format(g, feat_size, feat, weight, ground_truth_y)