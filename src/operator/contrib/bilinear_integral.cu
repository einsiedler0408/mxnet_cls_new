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
 * \file bilinear_integral.cu
 * \brief BilinearIntegral Operator
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
#include "./bilinear_integral-inl.h"

namespace mshadow {
namespace cuda {
namespace bilinear_integral {

static __device__ __forceinline__ bool between(float value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}

template <typename DType>
__global__ void BilinearIntegralForward(const int n, const bool out_zero, const int channels,
                                      const DType* input_data, const DType* input_offset,
                                      const int input_spatial_dim, 
                                      const int input_height, const int input_width,
                                      DType* output_data,
                                      const int output_spatial_dim,
                                      const int output_height, const int output_width) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int nc = index / output_spatial_dim;
    const int n  = nc / channels;
    const int c  = nc % channels;
	const int os = index % output_spatial_dim;
    const int oh = os / output_width;
    const int ow = os % output_width;

    float iw = ow + input_offset[n * 2 * output_spatial_dim + os];
    float ih = oh + input_offset[n * 2 * output_spatial_dim + os + output_spatial_dim];
    
    if (out_zero && (iw < 0 || iw > input_width-1 || ih < 0 || ih > input_height - 1))
        continue;
    
    int iw_round = round(iw);
    int ih_round = round(ih);
    int ihw_round = ih_round * input_width + iw_round;

    float a = 0.5 - (iw - iw_round);
    float b = 0.5 - (ih - ih_round);
    
    float a_square    = 0.5 * a * a;
    float a_1m_square = 0.5 * (1-a) * (1-a);
    float a_combine   = a * (1-a) + 0.5;
    
    float b_square    = 0.5 * b * b;
    float b_1m_square = 0.5 * (1-b) * (1-b);
    float b_combine   = b * (1-b) + 0.5;    
    
    const DType* input_data_cur = input_data + nc * input_spatial_dim;
      
    DType sum = 0;
    
    if (between(iw_round - 1, 0, input_width-1) && between(ih_round - 1, 0, input_height-1))
        sum += input_data_cur[ihw_round - input_width - 1] * a_square * b_square;
  
    if (between(iw_round    , 0, input_width-1) && between(ih_round - 1, 0, input_height-1))
        sum += input_data_cur[ihw_round - input_width    ] * a_combine * b_square;
  
    if (between(iw_round + 1, 0, input_width-1) && between(ih_round - 1, 0, input_height-1))
        sum += input_data_cur[ihw_round - input_width + 1] * a_1m_square * b_square;
    
    if (between(iw_round - 1, 0, input_width-1) && between(ih_round    , 0, input_height-1))
        sum += input_data_cur[ihw_round               - 1] * a_square * b_combine;
  
    if (between(iw_round    , 0, input_width-1) && between(ih_round    , 0, input_height-1))
        sum += input_data_cur[ihw_round                  ] * a_combine * b_combine;
  
    if (between(iw_round + 1, 0, input_width-1) && between(ih_round    , 0, input_height-1))
        sum += input_data_cur[ihw_round               + 1] * a_1m_square * b_combine;
  
    if (between(iw_round - 1, 0, input_width-1) && between(ih_round + 1, 0, input_height-1))
        sum += input_data_cur[ihw_round + input_width - 1] * a_square * b_1m_square;
  
    if (between(iw_round    , 0, input_width-1) && between(ih_round + 1, 0, input_height-1))
        sum += input_data_cur[ihw_round + input_width    ] * a_combine * b_1m_square;
  
    if (between(iw_round + 1, 0, input_width-1) && between(ih_round + 1, 0, input_height-1))
        sum += input_data_cur[ihw_round + input_width + 1] * a_1m_square * b_1m_square;


    output_data[index] += sum; 
  }
}

template <typename DType>
__global__ void BilinearIntegralBackward(const int n, const bool out_zero, const int channels,
                                      const DType* input_data, const DType* input_offset,
                                      const int input_spatial_dim, 
                                      const int input_height, const int input_width,
                                      const DType* output_grad,
                                      const int output_spatial_dim,
                                      const int output_height, const int output_width,
                                      DType* input_data_grad, DType* input_offset_grad) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int n = index / output_spatial_dim;
	const int os = index % output_spatial_dim;
    const int oh = os / output_width;
    const int ow = os % output_width;

    float iw = ow + input_offset[n * 2 * output_spatial_dim + os];
    float ih = oh + input_offset[n * 2 * output_spatial_dim + os + output_spatial_dim];
    
    if (out_zero && (iw < 0 || iw > input_width-1 || ih < 0 || ih > input_height - 1))
        continue;
    
    int iw_round = round(iw);
    int ih_round = round(ih);
    int ihw_round = ih_round * input_width + iw_round;

    float a = 0.5 - (iw - iw_round);
    float b = 0.5 - (ih - ih_round);
    
    float a_square    = 0.5 * a * a;
    float a_1m_square = 0.5 * (1-a) * (1-a);
    float a_combine   = a * (1-a) + 0.5;
    
    float b_square    = 0.5 * b * b;
    float b_1m_square = 0.5 * (1-b) * (1-b);
    float b_combine   = b * (1-b) + 0.5;    
    
    float da_square    = a;
    float da_1m_square = -(1-a);
    float da_combine   = 1 - 2*a;
    
    float db_square    = b;
    float db_1m_square = -(1-b);
    float db_combine   = 1 - 2*b; 
    
    for (int c = 0; c < channels; c++) {
        
        const DType output_grad_value = output_grad[(n * channels + c) * output_spatial_dim + os];
        const DType* input_data_cur   = input_data + (n * channels + c) * input_spatial_dim;
        DType* input_data_grad_cur    = input_data_grad + (n * channels + c) * input_spatial_dim;
          
        if (between(iw_round - 1, 0, input_width-1) && between(ih_round - 1, 0, input_height-1)){
            //sum += input_data_cur[ihw_round - input_width - 1] * a_square * b_square;
            atomicAdd(input_data_grad_cur + ihw_round - input_width - 1, output_grad_value * a_square * b_square);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round - input_width - 1] * da_square * b_square;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round - input_width - 1] * a_square * db_square;
        }
          
      
        if (between(iw_round    , 0, input_width-1) && between(ih_round - 1, 0, input_height-1)){
            //sum += input_data_cur[ihw_round - input_width    ] * a_combine * b_square;
            atomicAdd(input_data_grad_cur + ihw_round - input_width    , output_grad_value * a_combine * b_square);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round - input_width    ] * da_combine * b_square;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round - input_width    ] * a_combine * db_square;
        }
      
        if (between(iw_round + 1, 0, input_width-1) && between(ih_round - 1, 0, input_height-1)){
            //sum += input_data_cur[ihw_round - input_width + 1] * a_1m_square * b_square;
            atomicAdd(input_data_grad_cur + ihw_round - input_width + 1, output_grad_value * a_1m_square * b_square);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round - input_width + 1] * da_1m_square * b_square;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round - input_width + 1] * a_1m_square * db_square;
        }
    
        if (between(iw_round - 1, 0, input_width-1) && between(ih_round    , 0, input_height-1)){
            //sum += input_data_cur[ihw_round               - 1] * a_square * b_combine;
            atomicAdd(input_data_grad_cur + ihw_round               - 1, output_grad_value * a_square * b_combine);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round               - 1] * da_square * b_combine;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round               - 1] * a_square * db_combine;
        }
      
        if (between(iw_round    , 0, input_width-1) && between(ih_round    , 0, input_height-1)){
            //sum += input_data_cur[ihw_round                  ] * a_combine * b_combine;
            atomicAdd(input_data_grad_cur + ihw_round                  , output_grad_value * a_combine * b_combine);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round                  ] * da_combine * b_combine;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round                  ] * a_combine * db_combine;
        }
      
        if (between(iw_round + 1, 0, input_width-1) && between(ih_round    , 0, input_height-1)){
            //sum += input_data_cur[ihw_round               + 1] * a_1m_square * b_combine;
            atomicAdd(input_data_grad_cur + ihw_round               + 1, output_grad_value * a_1m_square * b_combine);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round               + 1] * da_1m_square * b_combine;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round               + 1] * a_1m_square * db_combine;
        }
      
        if (between(iw_round - 1, 0, input_width-1) && between(ih_round + 1, 0, input_height-1)){
            //sum += input_data_cur[ihw_round + input_width - 1] * a_square * b_1m_square;
            atomicAdd(input_data_grad_cur + ihw_round + input_width - 1, output_grad_value * a_square * b_1m_square);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round + input_width - 1] * da_square * b_1m_square;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round + input_width - 1] * a_square * db_1m_square;
        }
      
        if (between(iw_round    , 0, input_width-1) && between(ih_round + 1, 0, input_height-1)){
            //sum += input_data_cur[ihw_round + input_width    ] * a_combine * b_1m_square;
            atomicAdd(input_data_grad_cur + ihw_round + input_width    , output_grad_value * a_combine * b_1m_square);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round + input_width    ] * da_combine * b_1m_square;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round + input_width    ] * a_combine * db_1m_square;
        }
      
        if (between(iw_round + 1, 0, input_width-1) && between(ih_round + 1, 0, input_height-1)){
            //sum += input_data_cur[ihw_round + input_width + 1] * a_1m_square * b_1m_square;
            atomicAdd(input_data_grad_cur + ihw_round + input_width + 1, output_grad_value * a_1m_square * b_1m_square);
            input_offset_grad[n * 2 * output_spatial_dim + os]                      += - output_grad_value * input_data_cur[ihw_round + input_width + 1] * da_1m_square * b_1m_square;
            input_offset_grad[n * 2 * output_spatial_dim + os + output_spatial_dim] += - output_grad_value * input_data_cur[ihw_round + input_width + 1] * a_1m_square * db_1m_square;
        }

    }

  }
}

}  // namespace bilinear_integral
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class BilinearIntegralGPUOp : public Operator{
 public:
  explicit BilinearIntegralGPUOp(BilinearIntegralParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::bilinear_integral;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_data = in_data[bilinear_integral::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> input_offset = in_data[bilinear_integral::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_data = out_data[bilinear_integral::kOutput].get<xpu, 4, DType>(s);
    
    if (req[bilinear_integral::kOutput] == kWriteTo)
        output_data = 0;
    
    index_t batch_num = input_data.shape_[0];
    index_t channel_num = input_data.shape_[1];
    index_t input_height = input_data.shape_[2];
    index_t input_width = input_data.shape_[3];
    index_t output_height = output_data.shape_[2];
    index_t output_width = output_data.shape_[3];

    index_t num_kernels = batch_num * channel_num * output_height * output_width;
    using namespace mxnet_op;
    BilinearIntegralForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, param_.out_zero, channel_num,
          input_data.dptr_, input_offset.dptr_,
          input_height*input_width, input_height, input_width,
          output_data.dptr_,
          output_height*output_width, output_height, output_width);
    MSHADOW_CUDA_POST_KERNEL_CHECK(BilinearIntegralForward);
    
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
    using namespace mshadow::cuda::bilinear_integral;
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_data_grad = in_grad[bilinear_integral::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> input_offset_grad = in_grad[bilinear_integral::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_grad = out_grad[bilinear_integral::kOutput].get<xpu, 4, DType>(s);
    
    Tensor<xpu, 4, DType> input_data = in_data[bilinear_integral::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> input_offset = in_data[bilinear_integral::kOffset].get<xpu, 4, DType>(s);

    if (req[bilinear_integral::kData] == kWriteTo)
        input_data_grad = 0;  

    if (req[bilinear_integral::kOffset] == kWriteTo)
        input_offset_grad = 0;  
    
    index_t batch_num = input_data_grad.shape_[0];
    index_t channel_num = input_data_grad.shape_[1];
    index_t input_height = input_data_grad.shape_[2];
    index_t input_width = input_data_grad.shape_[3];
    index_t output_height = output_grad.shape_[2];
    index_t output_width = output_grad.shape_[3];

    index_t num_kernels = batch_num * output_height * output_width;
    using namespace mxnet_op;
    BilinearIntegralBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, param_.out_zero, channel_num,
          input_data.dptr_, input_offset.dptr_,
          input_height*input_width, input_height, input_width,
          output_grad.dptr_,
          output_height*output_width, output_height, output_width,
          input_data_grad.dptr_, input_offset_grad.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(BilinearIntegralBackward);
  }

 private:
  BilinearIntegralParam param_;
};  // class BilinearIntegralGPUOp

template<>
Operator* CreateOp<gpu>(BilinearIntegralParam param) {
  return new BilinearIntegralGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
