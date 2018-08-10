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
 * \file resize_crop.cu
 * \brief ResizeCrop Operator
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
#include "./resize_crop-inl.h"

namespace mshadow {
namespace cuda {
namespace resize_crop {

static __device__ __forceinline__ float triangleCoeff(float x)
{
    if (-1<=x && x<0) return x+1;
    if (0<=x && x<=1) return 1-x;
    return 0;
}


template <typename DType>
__global__ void ResizeCropForward(const int n, const float* random_params,
                                      const DType* input_data, const int channel_num,
                                      const int spatial_dim, const int height, const int width,
                                      DType* output_data, DType* output_offset, DType* output_mask) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int nc = index / spatial_dim;
    const int n = nc / channel_num;
    const int c = nc % channel_num;
	const int s = index % spatial_dim;
    const int oh = s / width;
    const int ow = s % width;

    float scale_h  = random_params[n * 4 + 0];
    float scale_w  = random_params[n * 4 + 1];
    float offset_h = random_params[n * 4 + 2];
    float offset_w = random_params[n * 4 + 3];
    
    float ih = (oh + offset_h) / scale_h;
    float iw = (ow + offset_w) / scale_w;
    
    if (ih >= 0 && ih <= height-1 && iw >= 0 && iw <= width-1) {
        output_offset[n * 2 * spatial_dim + s] = iw - ow;
        output_offset[n * 2 * spatial_dim + spatial_dim + s] = ih - oh;
        output_mask[n * spatial_dim + s] = 1;
    }  
    
    int ih_round = round(ih);
    int iw_round = round(iw);
    
    float kernel_radius_h = scale_h > 1 ? 1.0 : 1.0/scale_h;
    float alpha_h = 1.0f / kernel_radius_h;
    int radius_h = round(kernel_radius_h);
    
    float kernel_radius_w = scale_w > 1 ? 1.0 : 1.0/scale_w;
    float alpha_w = 1.0f / kernel_radius_w;
    int radius_w = round(kernel_radius_w);
    
    DType output_sum = 0;
    const DType* input_data_cur = input_data + nc * spatial_dim;
    
    for (int h = ih_round - radius_h; h <= ih_round + radius_h; h++) {
        for (int w = iw_round - radius_w; w <= iw_round + radius_w; w++) {
            if (h < 0 || h > height-1 || w < 0 || w > width-1)
                continue;
            DType input_value = input_data_cur[h * width + w];
            DType kernel_value = alpha_h * triangleCoeff(alpha_h * (ih - h)) * alpha_w * triangleCoeff(alpha_w * (iw - w));
            
            output_sum += input_value * kernel_value;
        }
    }
    
    output_data[index] = output_sum; 
  }
}

__global__ void ResizeCropRandom(const int n, const int height, const int width,
                       const float min_scale, const float max_scale, const bool keep_aspect_ratio,
                       float* random_params) {
    CUDA_KERNEL_LOOP(index, n) { 
        float scale_h, scale_w, offset_h, offset_w;
        scale_h = random_params[index * 4 + 0] * (max_scale - min_scale) + min_scale;
        scale_w = keep_aspect_ratio ? scale_h : random_params[index * 4 + 1] * (max_scale - min_scale) + min_scale;
        //scale_h = max(1., round(scale_h * height)) / float(height);
        //scale_w = max(1., round(scale_w * width)) / float(width);
        
        if (scale_h > 1){
            offset_h = random_params[index * 4 + 2] * height * (scale_h - 1); // rand_crop
        } else {
            offset_h = height * (scale_h - 1) / 2; // center_crop
        }
        
        if (scale_w > 1){
            offset_w = random_params[index * 4 + 3] * width * (scale_w - 1); // rand_crop
        } else {
            offset_w = width * (scale_w - 1) / 2; // center_crop
        }
        
        random_params[index * 4 + 0] = scale_h;
        random_params[index * 4 + 1] = scale_w;
        random_params[index * 4 + 2] = offset_h;
        random_params[index * 4 + 3] = offset_w;
    }
}


}  // namespace resize_crop
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ResizeCropGPUOp : public Operator{
 public:
  explicit ResizeCropGPUOp(ResizeCropParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::resize_crop;
    using namespace mxnet_op;

    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 4);
    CHECK_EQ(req.size(), 4);
    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> input_data = in_data[resize_crop::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_data = out_data[resize_crop::kOutput].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_offset = out_data[resize_crop::kOffset].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> output_mask = out_data[resize_crop::kMask].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> random_params = out_data[resize_crop::kRandomness].get<xpu, 4, DType>(s);
    
    index_t batch_num = input_data.shape_[0];
    index_t channel_num = input_data.shape_[1];
    index_t height = input_data.shape_[2];
    index_t width = input_data.shape_[3];
    
    CHECK_GE(param_.max_scale, param_.min_scale);
    CHECK_GT(param_.min_scale, 0);
    Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
    //Tensor<xpu, 1, float> random_params =
    //  ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(batch_num*4), s); // scale_h, scale_w, offset_h, offset_w
    prnd->SampleUniform(&random_params, 0, 1);
    
    ResizeCropRandom // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(batch_num), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (batch_num, height, width, param_.min_scale, param_.max_scale, param_.keep_aspect_ratio, random_params.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(ResizeCropRandom);
          
    index_t num_kernels = batch_num * channel_num * height * width;
    ResizeCropForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, random_params.dptr_, input_data.dptr_, channel_num, height * width, height, width,
          output_data.dptr_, output_offset.dptr_, output_mask.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(ResizeCropForward);
    
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
    using namespace mshadow::cuda::resize_crop;
    using namespace mxnet_op;

    CHECK_EQ(in_grad.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gdata = in_grad[resize_crop::kData].get<xpu, 4, real_t>(s);

    // can not assume the grad would be zero
    Assign(gdata, req[resize_crop::kData], 0);
  }

 private:
  ResizeCropParam param_;
};  // class ResizeCropGPUOp

template<>
Operator* CreateOp<gpu>(ResizeCropParam param) {
  return new ResizeCropGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
