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
 * \file downsample.cu
 * \brief Downsample Operator
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
#include "../../common/cuda_utils.h"
#include "./downsample-inl.h"

namespace mshadow {
namespace cuda {
namespace downsample {

template <typename DType>
__global__ void DownsampleForward(const int n,
                                      const DType* input_data, const int input_spatial_dim,
                                      const int input_height, const int input_width,
                                      const DType* kernel_data, const int kernel_size,
                                      DType* output_data, const int output_spatial_dim,
                                      const int output_height, const int output_width) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int bc = index / output_spatial_dim;
	const int s = index % output_spatial_dim;
    const int oh = s / output_width;
    const int ow = s % output_width;

    DType kernel_sum = 0;
    DType kernel_normalization = 0;
    const DType* input_data_cur = input_data + bc * input_spatial_dim;
    int kernel_dim = 2 * kernel_size + 1;
    for (int dh = -kernel_size; dh <= kernel_size; dh++) {
        for (int dw = -kernel_size; dw <= kernel_size; dw++) {
            int ih = oh * 2 + dh;
            int iw = ow * 2 + dw;
            int kh = kernel_size + dh;
            int kw = kernel_size + dw;
            if (ih < 0 || ih > input_height - 1 || iw < 0 || iw > input_width - 1)
                continue;
            
            DType input_value = input_data_cur[ih * input_width + iw];
            DType kernel_value = kernel_data[kh * kernel_dim + kw];
            
            kernel_sum += input_value * kernel_value;
            kernel_normalization += kernel_value;      
        }
    }
    output_data[index] += kernel_sum / kernel_normalization;
  }
}

template <typename DType>
__global__ void DownsampleBackward(const int n, 
                                      DType* input_grad, const int input_spatial_dim,
                                      const int input_height, const int input_width,
                                      const DType* kernel_data, const int kernel_size,
                                      const DType* output_grad, const int output_spatial_dim,
                                      const int output_height, const int output_width) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int bc = index / output_spatial_dim;
	const int s = index % output_spatial_dim;
    const int oh = s / output_width;
    const int ow = s % output_width;

    DType kernel_normalization = 0;
    int kernel_dim = 2 * kernel_size + 1;
    for (int dh = -kernel_size; dh <= kernel_size; dh++) {
        for (int dw = -kernel_size; dw <= kernel_size; dw++) {
            int ih = oh * 2 + dh;
            int iw = ow * 2 + dw;
            int kh = kernel_size + dh;
            int kw = kernel_size + dw;
            if (ih < 0 || ih > input_height - 1 || iw < 0 || iw > input_width - 1)
                continue;
            DType kernel_value = kernel_data[kh * kernel_dim + kw];
            kernel_normalization += kernel_value;      
        }
    }
    
    DType output_grad_value = output_grad[index];
    DType* input_grad_cur = input_grad + bc * input_spatial_dim;
    for (int dh = -kernel_size; dh <= kernel_size; dh++) {
        for (int dw = -kernel_size; dw <= kernel_size; dw++) {
            int ih = oh * 2 + dh;
            int iw = ow * 2 + dw;
            int kh = kernel_size + dh;
            int kw = kernel_size + dw;
            if (ih < 0 || ih > input_height - 1 || iw < 0 || iw > input_width - 1)
                continue;
            
            DType kernel_grad_value = kernel_data[kh * kernel_dim + kw] / kernel_normalization;
            atomicAdd(input_grad_cur + ih * input_width + iw, output_grad_value * kernel_grad_value);
        }
    }
  }
}

}  // namespace downsample
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class DownsampleGPUOp : public Operator{
 public:
  explicit DownsampleGPUOp(DownsampleParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::downsample;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 2);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_data = in_data[downsample::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> kernel_data = in_data[downsample::kKernel].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> output_data = out_data[downsample::kOutput].get<xpu, 4, DType>(s);
    if (req[downsample::kOutput] == kWriteTo)
        output_data = 0;
    
    index_t batch_num = input_data.shape_[0];
    index_t channel_num = input_data.shape_[1];
    index_t input_height = input_data.shape_[2];
    index_t input_width = input_data.shape_[3];
    index_t kernel_size = kernel_data.shape_[0] / 2;
    index_t output_height = output_data.shape_[2];
    index_t output_width = output_data.shape_[3];

    index_t num_kernels = batch_num * channel_num * output_height * output_width;
    using namespace mxnet_op;
    DownsampleForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, input_data.dptr_, input_height * input_width, input_height, input_width,
          kernel_data.dptr_, kernel_size, output_data.dptr_, output_height * output_width, output_height, output_width);
    MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleForward);
    
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
    using namespace mshadow::cuda::downsample;
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_grad = in_grad[downsample::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> kernel_grad = in_grad[downsample::kKernel].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> kernel_data = in_data[downsample::kKernel].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> output_grad = out_grad[downsample::kOutput].get<xpu, 4, DType>(s);

    if (req[downsample::kData] == kWriteTo)
        input_grad = 0;
    
    // we do not backward to kernel
    kernel_grad = 0;
    
    index_t batch_num = input_grad.shape_[0];
    index_t channel_num = input_grad.shape_[1];
    index_t input_height = input_grad.shape_[2];
    index_t input_width = input_grad.shape_[3];
    index_t kernel_size = kernel_data.shape_[0] / 2;
    index_t output_height = output_grad.shape_[2];
    index_t output_width = output_grad.shape_[3];

    index_t num_kernels = batch_num * channel_num * output_height * output_width;
    using namespace mxnet_op;
    DownsampleBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, input_grad.dptr_, input_height * input_width, input_height, input_width,
          kernel_data.dptr_, kernel_size, output_grad.dptr_, output_height * output_width, output_height, output_width);
    MSHADOW_CUDA_POST_KERNEL_CHECK(DownsampleBackward);
  }

 private:
  DownsampleParam param_;
};  // class DownsampleGPUOp

template<>
Operator* CreateOp<gpu>(DownsampleParam param) {
  return new DownsampleGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
