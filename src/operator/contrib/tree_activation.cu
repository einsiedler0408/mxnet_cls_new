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
 * Copyright (c) 2015 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file tree_activation.cu
 * \brief TreeActivation Operator
 * \author Shaoqing Ren, Xizhou Zhu, Jian Guo
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./tree_activation-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace tree_activation {

template <typename DType>
__global__ void TreeActivationForward(const int n, const DType* input_data,
                                      const int spatial_dim, const int tree_num, 
                                      const int tree_depth, const int node_num,
                                      DType* output_value, DType* output_index) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int s = index % spatial_dim;
	const int b = index / spatial_dim;
    const int tid = b % tree_num;

    const DType* input_data_cur = input_data + b * node_num * spatial_dim + s;
    DType* output_value_cur = output_value + b * tree_depth * spatial_dim + s;
    DType* output_index_cur = output_index + b * tree_depth * spatial_dim + s;
    
    output_value_cur[0] = 1;
    output_index_cur[0] = tid * node_num;
    int parent_node_id = 1;
    
    for (int i = 1; i < tree_depth; i++) {
        DType parent_data = input_data_cur[(parent_node_id - 1) * spatial_dim];
        if (parent_data > 0) {
            output_value_cur[i * spatial_dim] = parent_data;
            output_index_cur[i * spatial_dim] = tid * node_num + (parent_node_id * 2 - 1);
            parent_node_id = parent_node_id * 2;
        } else {
            output_value_cur[i * spatial_dim] = -parent_data;
            output_index_cur[i * spatial_dim] = tid * node_num + (parent_node_id * 2);
            parent_node_id = parent_node_id * 2 + 1;
        }
    }
    
    /*int parent_node_id = 1;
    DType parent_data = input_data_cur[(parent_node_id - 1) * spatial_dim];
    
    output_value_cur[0] = parent_data;
    output_index_cur[0] = tid * node_num + parent_node_id - 1;

    for (int i = 1; i < tree_depth; i++) {
        if (parent_data > 0) {
            parent_node_id = parent_node_id * 2;
            parent_data = input_data_cur[(parent_node_id - 1) * spatial_dim];
            output_value_cur[i * spatial_dim] = parent_data;
            output_index_cur[i * spatial_dim] = tid * node_num + parent_node_id - 1;           
        } else {
            parent_node_id = parent_node_id * 2 + 1;
            parent_data = input_data_cur[(parent_node_id - 1) * spatial_dim];
            output_value_cur[i * spatial_dim] = parent_data;
            output_index_cur[i * spatial_dim] = tid * node_num + parent_node_id - 1;    
        }
    }*/
  }
}

template <typename DType>
__global__ void TreeActivationBackward(const int n, DType* input_grad,
                                      const int spatial_dim, const int tree_num, 
                                      const int tree_depth, const int node_num,
                                      const DType* output_grad, const DType* output_index) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int s = index % spatial_dim;
	const int b = index / spatial_dim;

    DType* input_grad_cur = input_grad + b * node_num * spatial_dim + s;
    const DType* output_grad_cur = output_grad + b * tree_depth * spatial_dim + s;
    const DType* output_index_cur = output_index + b * tree_depth * spatial_dim + s;
    
    for (int i = 1; i < tree_depth; i++) {
        int current_node_id = int(output_index_cur[i * spatial_dim]) % node_num + 1;
        int parent_node_id = current_node_id / 2;
        
        if (current_node_id % 2 == 0) {
            input_grad_cur[(parent_node_id - 1) * spatial_dim] = output_grad_cur[i * spatial_dim];
        } else {
            input_grad_cur[(parent_node_id - 1) * spatial_dim] = -output_grad_cur[i * spatial_dim];
        }        
    }
    
    /*for (int i = 0; i < tree_depth; i++) {
        int current_node_id = int(output_index_cur[i * spatial_dim]) % node_num;
        input_grad_cur[current_node_id * spatial_dim] = output_grad_cur[i * spatial_dim];      
    }*/
  }
}

}  // namespace tree_activation
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class TreeActivationGPUOp : public Operator{
 public:
  explicit TreeActivationGPUOp(TreeActivationParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::tree_activation;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    CHECK_EQ(req.size(), 2);
    CHECK_EQ(req[treeAct::kValue], kWriteTo);
    CHECK_EQ(req[treeAct::kIndex], kWriteTo);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_data = in_data[treeAct::kData].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> output_value = out_data[treeAct::kValue].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_index = out_data[treeAct::kIndex].get<xpu, 4, DType>(s);

    index_t batch_num = input_data.shape_[0];
    index_t height = input_data.shape_[2];
    index_t width = input_data.shape_[3];
    
    index_t num_kernels = batch_num * param_.tree_num * height * width;
    using namespace mxnet_op;
    TreeActivationForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, input_data.dptr_, height * width, param_.tree_num, param_.tree_depth, (1 << param_.tree_depth) - 1, output_value.dptr_, output_index.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(TreeActivationForward);
    
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::tree_activation;
    CHECK_EQ(req[treeAct::kData], kWriteTo);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_grad = in_grad[treeAct::kData].get<xpu, 4, DType>(s);
    input_grad = 0;

    Tensor<xpu, 4, DType> output_grad = out_grad[treeAct::kValue].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_index = out_data[treeAct::kIndex].get<xpu, 4, DType>(s);

    index_t batch_num = input_grad.shape_[0];
    index_t height = input_grad.shape_[2];
    index_t width = input_grad.shape_[3];
    
    index_t num_kernels = batch_num * param_.tree_num * height * width;
    using namespace mxnet_op;
    TreeActivationBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, input_grad.dptr_, height * width, param_.tree_num, param_.tree_depth, (1 << param_.tree_depth) - 1, output_grad.dptr_, output_index.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(TreeActivationBackward);
  }

 private:
  TreeActivationParam param_;
};  // class TreeActivationGPUOp

template<>
Operator* CreateOp<gpu>(TreeActivationParam param) {
  return new TreeActivationGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
