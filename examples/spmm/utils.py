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

import dgl
from tvm.script import tir as T
from ogb.nodeproppred import DglNodePropPredDataset


def get_dataset(name: str):
    if name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
        g = arxiv[0][0]
    elif name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins")
        g = proteins[0][0]
    elif name == "products":
        products = DglNodePropPredDataset(name="ogbn-products")
        g = products[0][0]
    elif name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0]
    elif name == "citeseer":
        citeseer = dgl.data.CiteseerGraphDataset()
        g = citeseer[0]
    elif name == "cora":
        cora = dgl.data.CoraGraphDataset()
        g = cora[0]
    elif name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi)
    elif name == "reddit":
        reddit = dgl.data.RedditDataset()
        g = reddit[0]
    elif name == "dense":
        g = dgl.rand_graph(1024, 1024 * 1024 // 2)
    else:
        raise KeyError("Unknown dataset {}.".format(name))
    return g.int()


@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


@T.prim_func
def empty_func() -> None:
    T.evaluate(0)



@T.prim_func
def ell_inter_buffer(
    a: T.handle,
    a2: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    A2 = T.match_sparse_buffer(a2, (O, I, J), "float32")
    A_A = T.alloc_sparse_buffer((O, I, J), "float32",add_in_buffer_table= True)
    T.evaluate(0)
    
    
@T.prim_func
def ell_ind(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    m: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.dense_fixed(nnz_cols)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)



@T.prim_func
def ellfuse(
    a: T.handle,
    indptr_ii: T.handle,
    indices_ii: T.handle,
    indices_j: T.handle,
    # d1: T.int32,#tile_number
    d2: T.int32,#row_num
    # d3: T.int32,#tile_col_num
    nnz_tile_block: T.int32,
    nnz_rows: T.int32,#不同 tile block 的 nnz_row 需要加起来
    nnz_cols: T.int32,
) -> None:
    # d1 =T.var("int32")
    # d2 =T.var("int32")
    d3 =T.var("int32")
    # nnz_tile_block =T.var("int32")
    # nnz_rows =T.var("int32")
    R = T.dense_fixed(1, idtype="int32")
    IO = T.dense_fixed(nnz_tile_block, idtype="int32")#indptr_io内记录的是有多少个tile block具有非零项，indices_io记录的是含有非零项的tile block的索引
    II = T.sparse_variable(IO, (d2, nnz_rows), (indptr_ii, indices_ii), idtype="int32")#indptr_ii内记录的是当前tile block中含有多少个非零行，indices_ii记录的是非零行对应的索引值
    J = T.sparse_fixed(II, (d3, nnz_cols), indices_j, idtype="int32")#indices_jj记录的是当前非零行内非零列的索引值，确定唯一的非零元素
    A = T.match_sparse_buffer(a, (R, IO, II, J), dtype="float32")
    T.evaluate(0)