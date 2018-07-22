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
 * \file bilinear_downsample.cu
 * \brief BilinearDownsample Operator
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
#include "./bilinear_downsample-inl.h"

namespace mshadow {
namespace cuda {
namespace bilinear_downsample {

static __device__ __forceinline__ float triangleCoeff(float x)
{
    if (-1<=x && x<0) return x+1;
    if (0<=x && x<=1) return 1-x;
    return 0;
}

template <typename DType>
__global__ void BilinearDownsampleForward(const int n, float rescale, float kernel_radius,
                                      const DType* input_data, const int input_spatial_dim,
                                      const int input_height, const int input_width,
                                      DType* output_data, const int output_spatial_dim,
                                      const int output_height, const int output_width) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int bc = index / output_spatial_dim;
	const int s = index % output_spatial_dim;
    const int oh = s / output_width;
    const int ow = s % output_width;

    float ih = oh * rescale;
    float iw = ow * rescale;

    int ih_round = round(ih);
    int iw_round = round(iw);
    
    float alpha = 1.0f / kernel_radius;
    int radius = round(kernel_radius); // floor(kernel_radius + 0.5)
    
    DType output_sum = 0;
    DType kernel_sum = 0;
    const DType* input_data_cur = input_data + bc * input_spatial_dim;
    
    for (int h = ih_round - radius; h <= ih_round + radius; h++) {
        for (int w = iw_round - radius; w <= iw_round + radius; w++) {
            if (h < 0 || h > input_height-1 || w < 0 || w > input_width-1)
                continue;
            
            DType input_value = input_data_cur[h * input_width + w];
            DType kernel_value = alpha * triangleCoeff(alpha * (ih - h)) * alpha * triangleCoeff(alpha * (iw - w));
            
            output_sum += input_value * kernel_value;
            kernel_sum += kernel_value;
        }
    }
    
    output_data[index] += output_sum / kernel_sum; 
  }
}

template <typename DType>
__global__ void BilinearDownsampleBackward(const int n, float rescale, float kernel_radius,
                                      DType* input_grad, const int input_spatial_dim,
                                      const int input_height, const int input_width,
                                      const DType* output_grad, const int output_spatial_dim,
                                      const int output_height, const int output_width) {
  CUDA_KERNEL_LOOP(index, n) { 
	
    const int bc = index / output_spatial_dim;
	const int s = index % output_spatial_dim;
    const int oh = s / output_width;
    const int ow = s % output_width;

    float ih = oh * rescale;
    float iw = ow * rescale;

    int ih_round = round(ih);
    int iw_round = round(iw);
    
    float alpha = 1.0f / kernel_radius;
    int radius = round(kernel_radius); // floor(kernel_radius + 0.5)
    
    DType kernel_sum = 0;
    for (int h = ih_round - radius; h <= ih_round + radius; h++) {
        for (int w = iw_round - radius; w <= iw_round + radius; w++) {
            if (h < 0 || h > input_height-1 || w < 0 || w > input_width-1)
                continue;  
            DType kernel_value = alpha * triangleCoeff(alpha * (ih - h)) * alpha * triangleCoeff(alpha * (iw - w));
            kernel_sum += kernel_value;
        }
    }
    
    DType output_grad_value = output_grad[index];
    DType* input_grad_cur = input_grad + bc * input_spatial_dim;
    for (int h = ih_round - radius; h <= ih_round + radius; h++) {
        for (int w = iw_round - radius; w <= iw_round + radius; w++) {
            if (h < 0 || h > input_height-1 || w < 0 || w > input_width-1)
                continue;
            DType kernel_value = alpha * triangleCoeff(alpha * (ih - h)) * alpha * triangleCoeff(alpha * (iw - w));
            atomicAdd(input_grad_cur + h * input_width + w, output_grad_value * kernel_value / kernel_sum);
        }
    }
  }
}

}  // namespace bilinear_downsample
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class BilinearDownsampleGPUOp : public Operator{
 public:
  explicit BilinearDownsampleGPUOp(BilinearDownsampleParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::bilinear_downsample;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_data = in_data[bilinear_downsample::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_data = out_data[bilinear_downsample::kOutput].get<xpu, 4, DType>(s);
    if (req[bilinear_downsample::kOutput] == kWriteTo)
        output_data = 0;
    
    index_t batch_num = input_data.shape_[0];
    index_t channel_num = input_data.shape_[1];
    index_t input_height = input_data.shape_[2];
    index_t input_width = input_data.shape_[3];
    index_t output_height = output_data.shape_[2];
    index_t output_width = output_data.shape_[3];

    index_t num_kernels = batch_num * channel_num * output_height * output_width;
    using namespace mxnet_op;
    BilinearDownsampleForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, param_.rescale, param_.kernel_radius, input_data.dptr_, input_height * input_width, input_height, input_width,
          output_data.dptr_, output_height * output_width, output_height, output_width);
    MSHADOW_CUDA_POST_KERNEL_CHECK(BilinearDownsampleForward);
    
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
    using namespace mshadow::cuda::bilinear_downsample;
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_grad = in_grad[bilinear_downsample::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_grad = out_grad[bilinear_downsample::kOutput].get<xpu, 4, DType>(s);

    if (req[bilinear_downsample::kData] == kWriteTo)
        input_grad = 0;    
    
    index_t batch_num = input_grad.shape_[0];
    index_t channel_num = input_grad.shape_[1];
    index_t input_height = input_grad.shape_[2];
    index_t input_width = input_grad.shape_[3];
    index_t output_height = output_grad.shape_[2];
    index_t output_width = output_grad.shape_[3];

    index_t num_kernels = batch_num * channel_num * output_height * output_width;
    using namespace mxnet_op;
    BilinearDownsampleBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, param_.rescale, param_.kernel_radius, input_grad.dptr_, input_height * input_width, input_height, input_width,
          output_grad.dptr_, output_height * output_width, output_height, output_width);
    MSHADOW_CUDA_POST_KERNEL_CHECK(BilinearDownsampleBackward);
  }

 private:
  BilinearDownsampleParam param_;
};  // class BilinearDownsampleGPUOp

template<>
Operator* CreateOp<gpu>(BilinearDownsampleParam param) {
  return new BilinearDownsampleGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
