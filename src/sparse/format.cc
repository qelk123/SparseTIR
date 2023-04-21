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
      mask[part_id][bucket_id].push_back(1);//for float test
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
      ICHECK(static_cast<int>(col_indices[part_id][bucket_id].size()) == nnz * bucket_size)
          << "Padding error.";
      ICHECK(static_cast<int>(mask[part_id][bucket_id].size()) == nnz * bucket_size)
          << "Padding error.";
      NDArray row_indices_bucket_local = NDArray::Empty({nnz}, {kDLInt, 32, 1}, {kDLCPU, 0});
      NDArray col_indices_bucket_local =
          NDArray::Empty({nnz, bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
      NDArray mask_bucket_local = NDArray::Empty({nnz, bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
      if (nnz > 0) {
        row_indices_bucket_local.CopyFromBytes(row_indices[part_id][bucket_id].data(),
                                               nnz * sizeof(int));
        col_indices_bucket_local.CopyFromBytes(col_indices[part_id][bucket_id].data(),
                                               nnz * bucket_size * sizeof(int));
        mask_bucket_local.CopyFromBytes(mask[part_id][bucket_id].data(),
                                        nnz * bucket_size * sizeof(int));
      }
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




Array<Array<NDArray>> ColumnPartHyb2(int num_rows, int num_cols, NDArray indptr,
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
      degree_counter[part_id].insert(row_id);//part id对应的col part在row id对应的row上有非零值,row_id重复的次数就是该row在col part区间上的nnz
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
  Array<NDArray> tile_pos_nd;
  Array<NDArray> tile_indices_nd;
  Array<NDArray> row_pos_nd;
  Array<NDArray> row_indices_nd;
  Array<NDArray> col_indices_nd;
  Array<NDArray> mask_nd;
  Array<NDArray> nnz_row;

  std::vector<std::vector<int>> tile_pos(num_bkts);
  std::vector<std::vector<int>> tile_indices(num_bkts);
  std::vector<std::vector<int>> row_pos(num_bkts);
  std::vector<std::vector<int>> row_indices_r(num_bkts);
  std::vector<std::vector<int>> col_indices_r(num_bkts);
  std::vector<std::vector<int>> mask_r(num_bkts);
  std::vector<std::vector<int>> nnz_r(num_bkts);

  for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id) {
    tile_pos[bucket_id].resize(2);
    tile_pos[bucket_id][0]=0;
    tile_pos[bucket_id][1]=num_col_parts;
    tile_indices[bucket_id].resize(num_col_parts);
    row_pos[bucket_id].resize(num_col_parts+1);
    int row_nnz_cur = 0;
    for (int part_id = 0; part_id < num_col_parts; ++part_id) {
      row_pos[bucket_id][part_id] = row_nnz_cur;
      tile_indices[bucket_id][part_id]=part_id;
      int bucket_size = buckets_vec[bucket_id];
      // padding
      int remainder = col_indices[part_id][bucket_id].size() % bucket_size;
      if (remainder != 0) {
        for (int k = remainder; k < bucket_size; ++k) {
          col_indices[part_id][bucket_id].push_back(0);
          mask[part_id][bucket_id].push_back(0);
        }
      }
      row_nnz_cur+=row_indices[part_id][bucket_id].size();
      // int diffe_num_val=-1;
      // int diffe_num=0;
      // int count = true;
      for(auto i:row_indices[part_id][bucket_id]) {
        // if(bucket_id==num_bkts-1) std::cout<<"row_indices:"<<i<<"\n";
        // if(bucket_id==num_bkts-1 )  {
        //   if(i==diffe_num_val)
        //   {
        //     if(count)
        //     {
        //       diffe_num++;
        //       count =false;
        //     }
        //   }
        //   else{
        //     diffe_num_val = i;
        //     count = true;
        //   }
        //   std::cout<<"diff_num:"<<diffe_num<<"\n";
        // }
        row_indices_r[bucket_id].push_back(i);
      }
    }
    nnz_r[bucket_id].push_back(row_nnz_cur);
    col_indices_r[bucket_id].resize(row_nnz_cur*buckets_vec[bucket_id]);
    mask_r[bucket_id].resize(row_nnz_cur*buckets_vec[bucket_id]);
    int index_col_indices_r = 0;
    for(int part_id = 0; part_id < num_col_parts; ++part_id) {
      for(size_t j =0 ;j<col_indices[part_id][bucket_id].size();j++) {
        col_indices_r[bucket_id][index_col_indices_r] = col_indices[part_id][bucket_id][j];
        mask_r[bucket_id][index_col_indices_r] = mask[part_id][bucket_id][j];
        index_col_indices_r++;
      }
    }
    CHECK(index_col_indices_r==row_nnz_cur*buckets_vec[bucket_id])<<"nnz_number should equal to nnz_row_number*bucketsize";
    row_pos[bucket_id][num_col_parts] = row_nnz_cur;
  }

  for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id) {

    NDArray tile_pos_bucket_local = NDArray::Empty({2}, {kDLInt, 32, 1}, {kDLCPU, 0});
    NDArray tile_indices_bucket_local = NDArray::Empty({num_col_parts}, {kDLInt, 32, 1}, {kDLCPU, 0});
    NDArray row_pos_bucket_local = NDArray::Empty({num_col_parts+1}, {kDLInt, 32, 1}, {kDLCPU, 0});
    NDArray row_indices_r_bucket_local = NDArray::Empty({(int)(row_indices_r[bucket_id].size())}, {kDLInt, 32, 1}, {kDLCPU, 0});
    NDArray col_indices_r_bucket_local = NDArray::Empty({(int)(col_indices_r[bucket_id].size())}, {kDLInt, 32, 1}, {kDLCPU, 0});
    NDArray mask_r_bucket_local = NDArray::Empty({(int)(mask_r[bucket_id].size())}, {kDLInt, 32, 1}, {kDLCPU, 0});
    NDArray nnz_row_bucket_local = NDArray::Empty({(int)(nnz_r[bucket_id].size())}, {kDLInt, 32, 1}, {kDLCPU, 0});

    tile_pos_bucket_local.CopyFromBytes(tile_pos[bucket_id].data(),2 * sizeof(int));
    tile_indices_bucket_local.CopyFromBytes(tile_indices[bucket_id].data(),num_col_parts * sizeof(int));
    row_pos_bucket_local.CopyFromBytes(row_pos[bucket_id].data(),(num_col_parts+1) * sizeof(int));
    row_indices_r_bucket_local.CopyFromBytes(row_indices_r[bucket_id].data(),(row_indices_r[bucket_id].size()) * sizeof(int));
    col_indices_r_bucket_local.CopyFromBytes(col_indices_r[bucket_id].data(),(col_indices_r[bucket_id].size()) * sizeof(int));
    mask_r_bucket_local.CopyFromBytes(mask_r[bucket_id].data(),(mask_r[bucket_id].size()) * sizeof(int));
    nnz_row_bucket_local.CopyFromBytes(nnz_r[bucket_id].data(),(nnz_r[bucket_id].size()) * sizeof(int));


    tile_pos_nd.push_back(tile_pos_bucket_local);
    tile_indices_nd.push_back(tile_indices_bucket_local);
    row_pos_nd.push_back(row_pos_bucket_local);
    row_indices_nd.push_back(row_indices_r_bucket_local);
    col_indices_nd.push_back(col_indices_r_bucket_local);
    mask_nd.push_back(mask_r_bucket_local);
    nnz_row.push_back(nnz_row_bucket_local);
  }
  return {tile_pos_nd, tile_indices_nd, row_pos_nd, row_indices_nd, col_indices_nd, mask_nd, nnz_row};
}


Array<NDArray> ColumnELLReshapeIndex(int nv, int ne, int col_size, NDArray indptr,
                                           NDArray indices) {
  CHECK_EQ(indptr->dtype.bits, 32)
      << "Only support int32 index data type, got " << int(indptr->dtype.bits)
      << " bits for csf_indptr_0.";
  CHECK_EQ(indices->dtype.bits, 32)
      << "Only support int32 index data type, got " << int(indices->dtype.bits)
      << " bits for csf_indices_0.";
  CHECK_EQ(indptr->device.device_type, kDLCPU)
      << "Only support ColumnELLReshapeIndex conversion on CPU.";
  CHECK_EQ(indices->device.device_type, kDLCPU)
      << "Only support ColumnELLReshapeIndex conversion on CPU.";
  int * indptr_data = static_cast<int*>(indptr->data);
  int * indices_data = static_cast<int*>(indices->data);
  std::vector<int> ell_row_indices;
  std::vector<int> ell_col_indices;
  std::vector<int> ell_row_padding_num;
  int row_sum = 0;
  for (int row_index = 0; row_index < nv; row_index++) {
    int original_col_num = indptr_data[row_index+1] - indptr_data[row_index];
    int rows_for_original_row = (original_col_num + col_size - 1) / col_size;
    int src_start_index = indptr_data[row_index];
    row_sum += rows_for_original_row;
    ell_row_indices.resize(row_sum,row_index);
    ell_col_indices.resize(row_sum*col_size,0);
    int dst_start_index = (row_sum - rows_for_original_row) * col_size;
    ell_row_padding_num.push_back(rows_for_original_row * col_size - original_col_num);
    for (int i = 0; i < original_col_num; i++) {
      //TODO: use memcpy
      ell_col_indices[dst_start_index+i] = indices_data[src_start_index+i];
    }
  }
  //TODO: maybe transpose
  NDArray ell_col_indices_nd =
      NDArray::Empty({row_sum,col_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
  ell_col_indices_nd.CopyFromBytes(ell_col_indices.data(),
                                    row_sum * col_size * sizeof(int));
  NDArray ell_row_indices_nd =
      NDArray::Empty({row_sum}, {kDLInt, 32, 1}, {kDLCPU, 0});
  ell_row_indices_nd.CopyFromBytes(
      ell_row_indices.data(), row_sum * sizeof(int));

  NDArray ell_row_padding_num_nd =
      NDArray::Empty({nv}, {kDLInt, 32, 1}, {kDLCPU, 0});
  ell_row_padding_num_nd.CopyFromBytes(
      ell_row_padding_num.data(), nv * sizeof(int));

  return {ell_row_indices_nd,ell_col_indices_nd,ell_row_padding_num_nd};
}

Array<NDArray> ColumnEllDataPadding(int row_num,int nnz_col, Array<Integer> dim_k, NDArray indptr,
                                           Array<NDArray> or_buffer) {
  CHECK_EQ(indptr->dtype.bits, 32)
      << "Only support int32 index data type, got " << int(indptr->dtype.bits)
      << " bits for indptr.";
  CHECK_EQ(indptr->device.device_type, kDLCPU)
      << "Only support or_buffer padding conversion on CPU.";
  for (auto i: or_buffer) {
    CHECK_EQ(i->dtype.bits, 64)
        << "Only support double data type, got " << int(i->dtype.bits)
        << " bits for or_buffer.";
    CHECK_EQ(i->device.device_type, kDLCPU)
        << "Only support or_buffer padding on CPU.";
  }
  if(nnz_col == 1) {
    return or_buffer;
  }
  int * indptr_data = static_cast<int*>(indptr->data);
  int buffer_size = dim_k.size();
  double * value_data[buffer_size];
  double * padded_value_data[buffer_size];
  // std::vector<std::vector<double>> padded_buffer;
  std::vector<void *> padded_buffer_calloc;
  for (int i = 0; i < buffer_size; i++) {
    value_data[i] = static_cast<double*>(or_buffer[i]->data);
    // std::vector<double> item;
    // item.resize(row_num*nnz_col*dim_k[i],0);
    void * item_calloc = calloc(row_num*nnz_col*dim_k[i],sizeof(double));
    // padded_buffer.push_back(item);
    padded_buffer_calloc.push_back(item_calloc);
    // padded_value_data[i] = static_cast<double*>(padded_buffer.back().data());
    padded_value_data[i] = static_cast<double*>(item_calloc);
  }

  int row_num_after_padding = 0;
  for (int index = 0; index < indptr->shape[0]-1; index++) {
    for (int i = 0; i < buffer_size; i++) {
      std::memcpy((void*)(padded_value_data[i]+row_num_after_padding*nnz_col*dim_k[i]),
        (void*)(value_data[i]+indptr_data[index]*dim_k[i]),(indptr_data[index+1]-indptr_data[index])
        *dim_k[i]*sizeof(double));
    }
    row_num_after_padding += (indptr_data[index+1]-indptr_data[index]+nnz_col-1)/nnz_col;
    CHECK_LE(row_num_after_padding , row_num);
  }

  Array<NDArray> padded_buffer_nd_list;

  for (int i = 0; i < buffer_size; i++) {
    NDArray padded_buffer_nd = NDArray::Empty({row_num*nnz_col,dim_k[i]}, {kDLFloat, 64, 1}, {kDLCPU, 0});
    // padded_buffer_nd.CopyFromBytes(padded_buffer[i].data(),
    //                                         row_num*nnz_col*dim_k[i] * sizeof(double));
    padded_buffer_nd.CopyFromBytes(padded_buffer_calloc[i],
                                            row_num*nnz_col*dim_k[i] * sizeof(double));
    padded_buffer_nd_list.push_back(padded_buffer_nd);
  }

  return padded_buffer_nd_list;
}



/*!
 * \brief 3-Dimensional CSF to composable ELL format.
 * \param csf_indptr_0
 * \param csf_indices_0
 * \param csf_indptr_1
 * \param csf_indices_1
 * \param nnz_rows_bkt The number of nonzero rows parameter bucket (for output ELL3D format).
 * \param nnz_cols_bkt The number of nonzero cols parameter bucket (for output ELL3D format).
 * \return (indptr, row_indices, col_indices, mask), each one of them is a [num_buckets, *] array.
 */
Array<Array<NDArray>> CSFToELL3D(NDArray csf_indptr_0, NDArray csf_indices_0, NDArray csf_indptr_1,
                                 NDArray csf_indices_1, Array<Integer> nnz_rows_bkt,
                                 Array<Integer> nnz_cols_bkt) {
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
  CHECK_EQ(num_buckets, static_cast<int>(nnz_cols_bkt.size()))
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
    for (int i = csf_indptr_0_data[rel_id]; i < csf_indptr_0_data[rel_id + 1]; ++i) {
      int row = csf_indices_0_data[i];
      int num_cols_i = csf_indptr_1_data[i + 1] - csf_indptr_1_data[i];
      int bucket_id =
          std::upper_bound(nnz_cols_bkt_vec.begin(), nnz_cols_bkt_vec.end(), num_cols_i - 1) -
          nnz_cols_bkt_vec.begin();
      if (bucket_id == num_buckets) {
        bucket_id--;
      }
      int col_bucket_size = nnz_cols_bkt_vec[bucket_id];
      for (int j = csf_indptr_1_data[i]; j < csf_indptr_1_data[i + 1]; ++j) {
        int col = csf_indices_1_data[j];
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
  Array<NDArray> indptr_nd;
  Array<NDArray> row_indices_nd;
  Array<NDArray> col_indices_nd;
  Array<NDArray> mask_nd;
  for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id) {
    int row_bucket_size = nnz_rows_bkt_vec[bucket_id];
    int col_bucket_size = nnz_cols_bkt_vec[bucket_id];

    std::vector<int> indptr_bucket_local{0};
    std::vector<int> row_indices_bucket_local;
    std::vector<int> col_indices_bucket_local;
    std::vector<int> mask_bucket_local;
    for (int rel_id = 0; rel_id < num_rels; ++rel_id) {
      row_indices_bucket_local.insert(row_indices_bucket_local.end(),
                                      row_indices[bucket_id][rel_id].begin(),
                                      row_indices[bucket_id][rel_id].end());
      col_indices_bucket_local.insert(col_indices_bucket_local.end(),
                                      col_indices[bucket_id][rel_id].begin(),
                                      col_indices[bucket_id][rel_id].end());
      mask_bucket_local.insert(mask_bucket_local.end(), mask[bucket_id][rel_id].begin(),
                               mask[bucket_id][rel_id].end());
      int remainer_row = row_indices_bucket_local.size() % row_bucket_size;
      // padding
      if (remainer_row != 0) {
        for (int k = remainer_row; k < row_bucket_size; ++k) {
          row_indices_bucket_local.push_back(row_indices_bucket_local.back());
        }
      }
      int remainer_col = col_indices_bucket_local.size() % (row_bucket_size * col_bucket_size);
      if (remainer_col != 0) {
        for (int k = remainer_col; k < row_bucket_size * col_bucket_size; ++k) {
          col_indices_bucket_local.push_back(0);
          mask_bucket_local.push_back(0);
        }
      }
      indptr_bucket_local.push_back(row_indices_bucket_local.size() / row_bucket_size);
    }

    ICHECK((int)indptr_bucket_local.size() == (num_rels + 1)) << "Padding error.";
    NDArray indptr_bucket_local_nd = NDArray::Empty({num_rels + 1}, {kDLInt, 32, 1}, {kDLCPU, 0});
    indptr_bucket_local_nd.CopyFromBytes(indptr_bucket_local.data(), (num_rels + 1) * sizeof(int));
    int nnz = row_indices_bucket_local.size() / row_bucket_size;
    ICHECK((int)row_indices_bucket_local.size() == nnz * row_bucket_size) << "Padding error.";
    ICHECK((int)col_indices_bucket_local.size() == nnz * row_bucket_size * col_bucket_size)
        << "Padding error.";
    ICHECK((int)mask_bucket_local.size() == nnz * row_bucket_size * col_bucket_size)
        << "Padding error.";
    NDArray row_indices_bucket_local_nd =
        NDArray::Empty({nnz, row_bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
    row_indices_bucket_local_nd.CopyFromBytes(row_indices_bucket_local.data(),
                                              nnz * row_bucket_size * sizeof(int));
    NDArray col_indices_bucket_local_nd =
        NDArray::Empty({nnz, row_bucket_size, col_bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
    col_indices_bucket_local_nd.CopyFromBytes(
        col_indices_bucket_local.data(), nnz * row_bucket_size * col_bucket_size * sizeof(int));
    NDArray mask_bucket_local_nd =
        NDArray::Empty({nnz, row_bucket_size, col_bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
    mask_bucket_local_nd.CopyFromBytes(mask_bucket_local.data(),
                                       nnz * row_bucket_size * col_bucket_size * sizeof(int));
    indptr_nd.push_back(indptr_bucket_local_nd);
    row_indices_nd.push_back(row_indices_bucket_local_nd);
    col_indices_nd.push_back(col_indices_bucket_local_nd);
    mask_nd.push_back(mask_bucket_local_nd);
  }
  return {indptr_nd, row_indices_nd, col_indices_nd, mask_nd};
}

/*!
 * \brief Condense sparse matrix in CSR format to (t x 1) tiles, and group g tiles together.
 * \param indptr The indptr array of CSR format.
 * \param indices The indices array of CSR format.
 * \param t The tile size.
 * \param g The group size.
 * \return {group_indptr, tile_indices, mask}
 */
Array<NDArray> ConDense(NDArray indptr, NDArray indices, int t, int g) {
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
  // Condense matrix
  for (int row_tile_id = 0; row_tile_id < num_tiles; ++row_tile_id) {
    int tile_begin_row = row_tile_id * t;
    int tile_end_row = std::min(tile_begin_row + t, n);
    for (int i = tile_begin_row; i < tile_end_row; ++i) {
      for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
        int row = i;
        int col = indices_data[j];
        col_row_map.insert({col, row});//处于某一个tile区间内的index值（包括了col值和row值）
      }
    }

    int tile_counter = 0;
    for (auto unique_col_itr = col_row_map.begin(); unique_col_itr != col_row_map.end();) {
      int col = unique_col_itr->first;
      auto eq_range = col_row_map.equal_range(unique_col_itr->first);
      // int nnz_inside_tile = 0;
      // for (auto equal_iter = eq_range.first; equal_iter != eq_range.second; ++equal_iter) {
      //   nnz_inside_tile++;
      // }
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

      unique_col_itr = eq_range.second;
    }
    // update group indptr
    group_indptr.push_back(nnz_groups);
    // clear col-row multimap
    col_row_map.clear();
  }

  // Convert to NDArray
  NDArray group_indptr_nd = NDArray::Empty({num_tiles + 1}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray tile_indices_nd = NDArray::Empty({nnz_groups, g}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray mask_nd = NDArray::Empty({nnz_groups, t, g}, {kDLInt, 32, 1}, {kDLCPU, 0});
  group_indptr_nd.CopyFromBytes(group_indptr.data(), (num_tiles + 1) * sizeof(int));
  tile_indices_nd.CopyFromBytes(tile_indices.data(), (nnz_groups * g) * sizeof(int));
  mask_nd.CopyFromBytes(mask.data(), (nnz_groups * t * g) * sizeof(int));
  return {group_indptr_nd, tile_indices_nd, mask_nd};
}

namespace sparse {
TVM_REGISTER_GLOBAL("tir.sparse.ColumnPartHyb").set_body_typed(ColumnPartHyb);
TVM_REGISTER_GLOBAL("tir.sparse.ColumnPartHyb2").set_body_typed(ColumnPartHyb2);
TVM_REGISTER_GLOBAL("tir.sparse.ColumnELLReshapeIndex").set_body_typed(ColumnELLReshapeIndex);
TVM_REGISTER_GLOBAL("tir.sparse.ConDense").set_body_typed(ConDense);
TVM_REGISTER_GLOBAL("tir.sparse.CSFToELL3D").set_body_typed(CSFToELL3D);
TVM_REGISTER_GLOBAL("tir.sparse.ColumnEllDataPadding").set_body_typed(ColumnEllDataPadding);
}  // namespace sparse
}  // namespace tvm
