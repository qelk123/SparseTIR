/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file format.cc
 * \brief Sparse format conversion routines.
 */

#include <tvm/ir/expr.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

namespace tvm {

using runtime::NDArray;

/*!
 * \brief Partition input CSR matrix by columns and collect rows into buckets according to non zero
 * elements per row.
 * \param num_rows Number of rows in the CSR matrix.
 * \param num_cols Number of columns in the CSR matrix.
 * \param indptr The indptr array of CSR matrix.
 * \param indices The indices array of CSR matrix.
 * \param num_col_parts Number of column partitions.
 * \param buckets The bucket sizes array.
 * \return {row_indices, col_indices, mask}, each one of them is a [num_col_parts, num_buckets *]
 * array.
 */
Array<Array<Array<NDArray>>> ColumnPartHyb(int num_rows, int num_cols, NDArray indptr,
                                           NDArray indices, int num_col_parts,
                                           Array<Integer> buckets) {
  int partition_size = (num_cols + num_col_parts - 1) / num_col_parts;
  int num_bkts = buckets.size();
  std::vector<int> buckets_vec;
  for (const Integer& bucket_size : buckets) {
    buckets_vec.push_back(bucket_size->value);
  }

  CHECK_EQ(indptr->dtype.bits, 32) << "Only support int32 index data type, got "
                                   << int(indptr->dtype.bits) << " bits for indptr.";
  CHECK_EQ(indices->dtype.bits, 32) << "Only support int32 index data type, got "
                                    << int(indices->dtype.bits) << " bits for indices.";
  CHECK_EQ(indptr->device.device_type, kDLCPU) << "Only support ColumnPartHyb conversion on CPU.";
  CHECK_EQ(indices->device.device_type, kDLCPU) << "Only support ColumnPartHyb conversion on CPU.";
  int* indptr_data = static_cast<int*>(indptr->data);
  int* indices_data = static_cast<int*>(indices->data);
  std::vector<std::unordered_multiset<int>> degree_counter(num_col_parts);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      int row_id = i;
      int col_id = indices_data[j];
      int part_id = col_id / partition_size;
      degree_counter[part_id].insert(row_id);
    }
  }

  /* (num_parts, num_buckets, ...) */
  std::vector<std::vector<std::vector<int>>> row_indices(num_col_parts);
  std::vector<std::vector<std::vector<int>>> col_indices(num_col_parts);
  std::vector<std::vector<std::vector<int>>> mask(num_col_parts);
  // init row_indices, col_indices, mask
  for (int part_id = 0; part_id < num_col_parts; ++part_id) {
    for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id) {
      row_indices[part_id].push_back(std::vector<int>());
      col_indices[part_id].push_back(std::vector<int>());
      mask[part_id].push_back(std::vector<int>());
    }
  }
  for (int i = 0; i < num_rows; ++i) {
    for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      int row_id = i;
      int col_id = indices_data[j];
      int part_id = col_id / partition_size;
      int degree = degree_counter[part_id].count(row_id);
      int bucket_id = std::upper_bound(buckets_vec.begin(), buckets_vec.end(), degree - 1) -
                      buckets_vec.begin();
      if (bucket_id == num_bkts) {
        bucket_id--;
      }
      int bucket_size = buckets_vec[bucket_id];
      bool create_new_bucket = false;
      int remainder = col_indices[part_id][bucket_id].size() % bucket_size;
      if (remainder != 0) {
        ICHECK(!row_indices[part_id][bucket_id].empty()) << "row indices should not be empty.";
        if (row_id != row_indices[part_id][bucket_id].back()) {
          // padding
          for (int k = remainder; k < bucket_size; ++k) {
            col_indices[part_id][bucket_id].push_back(0);
            mask[part_id][bucket_id].push_back(0);
          }
          create_new_bucket = true;
        }
      } else {
        create_new_bucket = true;
      }
      if (create_new_bucket) {
        ICHECK(col_indices[part_id][bucket_id].size() % bucket_size == 0) << "Invalid padding";
        row_indices[part_id][bucket_id].push_back(row_id);
      }
      col_indices[part_id][bucket_id].push_back(col_id);
      mask[part_id][bucket_id].push_back(1);
    }
  }

  // final padding and conversion to NDArray
  Array<Array<NDArray>> row_indices_nd;
  Array<Array<NDArray>> col_indices_nd;
  Array<Array<NDArray>> mask_nd;
  for (int part_id = 0; part_id < num_col_parts; ++part_id) {
    Array<NDArray> row_indices_part_local;
    Array<NDArray> col_indices_part_local;
    Array<NDArray> mask_part_local;
    for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id) {
      int bucket_size = buckets_vec[bucket_id];
      // padding
      int remainder = col_indices[part_id][bucket_id].size() % bucket_size;
      if (remainder != 0) {
        for (int k = remainder; k < bucket_size; ++k) {
          col_indices[part_id][bucket_id].push_back(0);
          mask[part_id][bucket_id].push_back(0);
        }
      }
      // conversion to NDArray
      int nnz = row_indices[part_id][bucket_id].size();
      ICHECK(int(col_indices[part_id][bucket_id].size()) == nnz * bucket_size) << "Padding error.";
      ICHECK(int(mask[part_id][bucket_id].size()) == nnz * bucket_size) << "Padding error.";
      NDArray row_indices_bucket_local = NDArray::Empty({nnz}, {kDLInt, 32, 1}, {kDLCPU, 0});
      NDArray col_indices_bucket_local =
          NDArray::Empty({nnz, bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
      NDArray mask_bucket_local = NDArray::Empty({nnz, bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
      row_indices_bucket_local.CopyFromBytes(row_indices[part_id][bucket_id].data(),
                                             nnz * sizeof(int));
      col_indices_bucket_local.CopyFromBytes(col_indices[part_id][bucket_id].data(),
                                             nnz * bucket_size * sizeof(int));
      mask_bucket_local.CopyFromBytes(mask[part_id][bucket_id].data(),
                                      nnz * bucket_size * sizeof(int));
      row_indices_part_local.push_back(row_indices_bucket_local);
      col_indices_part_local.push_back(col_indices_bucket_local);
      mask_part_local.push_back(mask_bucket_local);
    }
    row_indices_nd.push_back(row_indices_part_local);
    col_indices_nd.push_back(col_indices_part_local);
    mask_nd.push_back(mask_part_local);
  }

  return {row_indices_nd, col_indices_nd, mask_nd};
}

/*!
 * \brief HeteroCSR to HeteroELL format.
 * \param indptr_arr The indptr array of CSR for each relation.
 * \param indices_arr The indices array of CSR for each relation.
 * \param nnz_rows_bkt The number of nonzero rows parameter bucket (for output ELL3D format).
 * \param nnz_cols_bkt The number of nonzero cols parameter bucket (for output ELL3D format).
 * \return (row_indices, col_indices, mask), each one of them is a [num_buckets, num_rels, *] array.
 */
Array<Array<Array<NDArray>>> CSFToELL3D(NDArray csf_indptr_0, NDArray csf_indices_0,
                                        NDArray csf_indptr_1, NDArray csf_indices_1,
                                        Array<Integer> nnz_rows_bkt, Array<Integer> nnz_cols_bkt) {
  CHECK_EQ(csf_indptr_0->dtype.bits, 32)
      << "Only support int32 index data type, got " << int(csf_indptr_0->dtype.bits)
      << " bits for csf_indptr_0.";
  CHECK_EQ(csf_indices_0->dtype.bits, 32)
      << "Only support int32 index data type, got " << int(csf_indices_0->dtype.bits)
      << " bits for csf_indices_0.";
  CHECK_EQ(csf_indptr_1->dtype.bits, 32)
      << "Only support int32 index data type, got " << int(csf_indptr_1->dtype.bits)
      << " bits for csf_indptr_0.";
  CHECK_EQ(csf_indices_1->dtype.bits, 32)
      << "Only support int32 index data type, got " << int(csf_indices_1->dtype.bits)
      << " bits for csf_indices_0.";
  CHECK_EQ(csf_indptr_0->device.device_type, kDLCPU)
      << "Only support CSFToELL3D conversion on CPU.";
  CHECK_EQ(csf_indices_0->device.device_type, kDLCPU)
      << "Only support CSFToeLL3D conversion on CPU.";
  CHECK_EQ(csf_indptr_1->device.device_type, kDLCPU)
      << "Only support CSFToELL3D conversion on CPU.";
  CHECK_EQ(csf_indices_1->device.device_type, kDLCPU)
      << "Only support CSFToeLL3D conversion on CPU.";

  int num_rels = csf_indptr_0->shape[0] - 1;
  int num_buckets = nnz_rows_bkt.size();
  CHECK_EQ(num_buckets, int(nnz_cols_bkt.size()))
      << "Input nnz_rows and nnz_cols should have same length.";
  std::vector<int> nnz_rows_bkt_vec, nnz_cols_bkt_vec;
  for (const Integer& nnz_rows : nnz_rows_bkt) {
    nnz_rows_bkt_vec.push_back(nnz_rows->value);
  }
  for (const Integer& nnz_cols : nnz_cols_bkt) {
    nnz_cols_bkt_vec.push_back(nnz_cols->value);
  }

  for (size_t i = 1; i < nnz_cols_bkt_vec.size(); ++i) {
    CHECK_LT(nnz_cols_bkt_vec[i - 1], nnz_cols_bkt_vec[i])
        << "The given nnz_cols_bkt should be ascending.";
  }

  /* (num_buckets, num_rels) */
  std::vector<std::vector<std::vector<int>>> row_indices(num_buckets);
  std::vector<std::vector<std::vector<int>>> col_indices(num_buckets);
  std::vector<std::vector<std::vector<int>>> mask(num_buckets);
  // init row_indices, col_indices, mask
  for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id) {
    for (int rel_id = 0; rel_id < num_rels; ++rel_id) {
      row_indices[bucket_id].push_back(std::vector<int>());
      col_indices[bucket_id].push_back(std::vector<int>());
      mask[bucket_id].push_back(std::vector<int>());
    }
  }

  int* csf_indptr_0_data = static_cast<int*>(csf_indptr_0->data);
  int* csf_indices_0_data = static_cast<int*>(csf_indices_0->data);
  int* csf_indptr_1_data = static_cast<int*>(csf_indptr_1->data);
  int* csf_indices_1_data = static_cast<int*>(csf_indices_1->data);

  for (int rel_id = 0; rel_id < num_rels; ++rel_id) {
    int num_rows = csf_indptr_0_data[rel_id + 1] - csf_indptr_0_data[rel_id];
    int* indptr_1_data = csf_indptr_1_data + csf_indptr_0_data[rel_id + 1];
    int* indices_0_data = csf_indices_0_data + csf_indptr_0_data[rel_id + 1];
    for (int i = 0; i < num_rows; ++i) {
      int row = indices_0_data[i];
      int num_cols_i = indptr_1_data[i + 1] - indptr_1_data[i];
      int bucket_id =
          std::upper_bound(nnz_cols_bkt_vec.begin(), nnz_cols_bkt_vec.end(), num_cols_i - 1) -
          nnz_cols_bkt_vec.begin();
      if (bucket_id == num_buckets) {
        bucket_id--;
      }
      int col_bucket_size = nnz_cols_bkt_vec[bucket_id];
      int* indices_1_data = csf_indices_1_data + indptr_1_data[i];
      for (int j = indptr_1_data[i]; j < indptr_1_data[i + 1]; ++j) {
        int col = indices_1_data[j];
        int remainder = col_indices[bucket_id][rel_id].size() % col_bucket_size;
        bool create_new_bucket = false;
        if (remainder != 0) {
          ICHECK(!row_indices[bucket_id][rel_id].empty()) << "row indices should not be empty.";
          if (row != row_indices[bucket_id][rel_id].back()) {
            // padding
            for (int k = remainder; k < col_bucket_size; ++k) {
              col_indices[bucket_id][rel_id].push_back(0);
              mask[bucket_id][rel_id].push_back(0);
            }
            create_new_bucket = true;
          }
        } else {
          create_new_bucket = true;
        }
        if (create_new_bucket) {
          ICHECK(col_indices[bucket_id][rel_id].size() % col_bucket_size == 0) << "Invalid padding";
          row_indices[bucket_id][rel_id].push_back(row);
        }
        col_indices[bucket_id][rel_id].push_back(col);
        mask[bucket_id][rel_id].push_back(1);
      }
    }
  }

  // final padding and conversion to NDArray
  Array<Array<NDArray>> row_indices_nd;
  Array<Array<NDArray>> col_indices_nd;
  Array<Array<NDArray>> mask_nd;
  for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id) {
    Array<NDArray> row_indices_bucket_local;
    Array<NDArray> col_indices_bucket_local;
    Array<NDArray> mask_bucket_local;
    int row_bucket_size = nnz_rows_bkt_vec[bucket_id];
    int col_bucket_size = nnz_cols_bkt_vec[bucket_id];
    for (int rel_id = 0; rel_id < num_rels; ++rel_id) {
      int remainer_row = row_indices[bucket_id][rel_id].size() % row_bucket_size;
      // padding
      if (remainer_row != 0) {
        for (int k = remainer_row; k < row_bucket_size; ++k) {
          row_indices[bucket_id][rel_id].push_back(row_indices[bucket_id][rel_id].back());
        }
      }
      int remainer_col =
          col_indices[bucket_id][rel_id].size() % (row_bucket_size * col_bucket_size);
      if (remainer_col != 0) {
        for (int k = remainer_col; k < row_bucket_size * col_bucket_size; ++k) {
          col_indices[bucket_id][rel_id].push_back(0);
          mask[bucket_id][rel_id].push_back(0);
        }
      }
      // conversion to NDArray
      int nnz_buckets = row_indices[bucket_id][rel_id].size() / row_bucket_size;
      ICHECK(int(row_indices[bucket_id][rel_id].size()) == nnz_buckets * row_bucket_size)
          << "Padding error.";
      ICHECK(int(col_indices[bucket_id][rel_id].size()) ==
             nnz_buckets * row_bucket_size * col_bucket_size)
          << "Padding error.";
      ICHECK(int(mask[bucket_id][rel_id].size()) == nnz_buckets * row_bucket_size * col_bucket_size)
          << "Padding error.";
      NDArray row_indices_rel_local =
          NDArray::Empty({nnz_buckets, row_bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
      NDArray col_indices_rel_local = NDArray::Empty(
          {nnz_buckets, row_bucket_size, col_bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
      NDArray mask_rel_local = NDArray::Empty({nnz_buckets, row_bucket_size, col_bucket_size},
                                              {kDLInt, 32, 1}, {kDLCPU, 0});
      row_indices_rel_local.CopyFromBytes(row_indices[bucket_id][rel_id].data(),
                                          nnz_buckets * row_bucket_size * sizeof(int));
      col_indices_rel_local.CopyFromBytes(
          col_indices[bucket_id][rel_id].data(),
          nnz_buckets * row_bucket_size * col_bucket_size * sizeof(int));
      mask_rel_local.CopyFromBytes(mask[bucket_id][rel_id].data(),
                                   nnz_buckets * row_bucket_size * col_bucket_size * sizeof(int));
      row_indices_bucket_local.push_back(row_indices_rel_local);
      col_indices_bucket_local.push_back(col_indices_rel_local);
      mask_bucket_local.push_back(mask_rel_local);
    }
    row_indices_nd.push_back(row_indices_bucket_local);
    col_indices_nd.push_back(col_indices_bucket_local);
    mask_nd.push_back(mask_bucket_local);
  }
  return {row_indices_nd, col_indices_nd, mask_nd};
}

/*!
 * \brief Condense sparse matrix in CSR format to (t x 1) tiles, and group g tiles together.
 * if nonzero elements in a tile is less then threshold, collect them to another DCSR format.
 * \param indptr The indptr array of CSR format.
 * \param indices The indices array of CSR format.
 * \param t The tile size.
 * \param g The group size.
 * \param threshold The threshold for determining whether a tile should be collected in tile format
 * or DCSR format.
 * \return {group_indptr, tile_indices, mask, dcsr_row_indices, dcsr_indptr, dcsr_col_indices}
 */
Array<NDArray> ConDense(NDArray indptr, NDArray indices, int t, int g, int threshold) {
  // Check inputs
  CHECK_EQ(indptr->dtype.bits, 32) << "Only support int32 index data type, got "
                                   << int(indptr->dtype.bits) << " bits for indptr.";
  CHECK_EQ(indices->dtype.bits, 32) << "Only support int32 index data type, got "
                                    << int(indices->dtype.bits) << " bits for indices.";
  CHECK_EQ(indptr->device.device_type, kDLCPU) << "Only support ConDense conversion on CPU.";
  CHECK_EQ(indices->device.device_type, kDLCPU) << "Only support ConDense conversion on CPU.";
  // Get data from NDArrays
  int* indptr_data = static_cast<int*>(indptr->data);
  int* indices_data = static_cast<int*>(indices->data);
  // Set up return values
  int n = indptr->shape[0] - 1;
  int num_tiles = (n + t - 1) / t;
  int nnz_groups = 0;
  std::vector<int> group_indptr;
  group_indptr.reserve(num_tiles + 1);
  std::vector<int> tile_indices;
  std::vector<int> mask;
  group_indptr.push_back(0);
  std::multimap<int, int> col_row_map;
  std::multimap<int, int> dcsr_row_col_map;
  // Condense matrix
  for (int row_tile_id = 0; row_tile_id < num_tiles; ++row_tile_id) {
    int tile_begin_row = row_tile_id * t;
    int tile_end_row = std::min(tile_begin_row + t, n);
    for (int i = tile_begin_row; i < tile_end_row; ++i) {
      for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
        int row = i;
        int col = indices_data[j];
        col_row_map.insert({col, row});
      }
    }

    int tile_counter = 0;
    for (auto unique_col_itr = col_row_map.begin(); unique_col_itr != col_row_map.end();) {
      int col = unique_col_itr->first;
      auto eq_range = col_row_map.equal_range(unique_col_itr->first);
      int nnz_inside_tile = 0;
      for (auto equal_iter = eq_range.first; equal_iter != eq_range.second; ++equal_iter) {
        nnz_inside_tile++;
      }
      if (nnz_inside_tile >= threshold) {
        // add tile to blockized format.
        tile_counter++;
        // new group
        if (tile_counter == 1) {
          nnz_groups++;
          tile_indices.resize(nnz_groups * g, 0);
          mask.resize(nnz_groups * t * g, 0);
        }
        // update tile_indices and mask
        tile_indices[(nnz_groups - 1) * g + (tile_counter - 1)] = col;
        for (auto equal_itr = eq_range.first; equal_itr != eq_range.second; ++equal_itr) {
          int row_local = equal_itr->second - tile_begin_row;
          mask[(nnz_groups - 1) * t * g + row_local * g + (tile_counter - 1)] = 1;
        }
        // reset tile_counter
        if (tile_counter == g) {
          tile_counter = 0;
        }
      } else {
        // add tile to dscr format.
        int row = eq_range.first->second;
        dcsr_row_col_map.insert({row, col});
      }
      unique_col_itr = eq_range.second;
    }
    // update group indptr
    group_indptr.push_back(nnz_groups);
    // clear col-row multimap
    col_row_map.clear();
  }

  // Remaining DCSR format.
  std::vector<int> dcsr_row_indices;
  std::vector<int> dcsr_col_indices;
  std::vector<int> dcsr_indptr{0};
  int offset = 0;
  for (auto unique_row_itr = dcsr_row_col_map.begin(); unique_row_itr != dcsr_row_col_map.end();) {
    int row = unique_row_itr->first;
    dcsr_row_indices.push_back(row);
    auto eq_range = dcsr_row_col_map.equal_range(unique_row_itr->first);
    for (auto equal_itr = eq_range.first; equal_itr != eq_range.second; ++equal_itr) {
      offset++;
      int col = equal_itr->second;
      dcsr_col_indices.push_back(col);
    }
    dcsr_indptr.push_back(offset);
    unique_row_itr = eq_range.second;
  }
  int dcsr_num_rows = dcsr_row_indices.size();
  int dcsr_nnz = dcsr_col_indices.size();
  ICHECK((int)dcsr_indptr.size() == dcsr_num_rows + 1) << "dcsr indptr size should be num_rows + 1";

  // Convert to NDArray
  NDArray group_indptr_nd = NDArray::Empty({num_tiles + 1}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray tile_indices_nd = NDArray::Empty({nnz_groups, g}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray mask_nd = NDArray::Empty({nnz_groups, t, g}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray dcsr_row_indices_nd = NDArray::Empty({dcsr_num_rows}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray dcsr_indptr_nd = NDArray::Empty({dcsr_num_rows + 1}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray dcsr_col_indices_nd = NDArray::Empty({dcsr_nnz}, {kDLInt, 32, 1}, {kDLCPU, 0});
  group_indptr_nd.CopyFromBytes(group_indptr.data(), (num_tiles + 1) * sizeof(int));
  tile_indices_nd.CopyFromBytes(tile_indices.data(), (nnz_groups * g) * sizeof(int));
  mask_nd.CopyFromBytes(mask.data(), (nnz_groups * t * g) * sizeof(int));
  if (dcsr_num_rows > 0) {
    dcsr_row_indices_nd.CopyFromBytes(dcsr_row_indices.data(), dcsr_num_rows * sizeof(int));
  }
  dcsr_indptr_nd.CopyFromBytes(dcsr_indptr.data(), (dcsr_num_rows + 1) * sizeof(int));
  if (dcsr_nnz > 0) {
    dcsr_col_indices_nd.CopyFromBytes(dcsr_col_indices.data(), dcsr_nnz * sizeof(int));
  }
  return {group_indptr_nd,     tile_indices_nd, mask_nd,
          dcsr_row_indices_nd, dcsr_indptr_nd,  dcsr_col_indices_nd};
}

namespace sparse {
TVM_REGISTER_GLOBAL("tir.sparse.ColumnPartHyb").set_body_typed(ColumnPartHyb);
TVM_REGISTER_GLOBAL("tir.sparse.ConDense").set_body_typed(ConDense);
TVM_REGISTER_GLOBAL("tir.sparse.CSFToELL3D").set_body_typed(CSFToELL3D);
}  // namespace sparse
}  // namespace tvm