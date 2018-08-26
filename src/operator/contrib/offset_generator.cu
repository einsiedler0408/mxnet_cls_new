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
 * \file offset_generator.cu
 * \brief OffsetGenerator Operator
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
#include "./offset_generator-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace offset_generator {

//[ a0 + b0 + a0*b0 + a3*b1, a1 + b1 + a1*b0 + a4*b1, a2 + b2 + a2*b0 + a5*b1 ]
//[ a3 + b3 + a0*b3 + a3*b4, a4 + b4 + a1*b3 + a4*b4, a5 + b5 + a2*b3 + a5*b4 ]
template <typename DType>
__global__ void AffineCompositeForward(const int n, const DType* data1, const DType* data2,
                                      const int spatial_dim, const int channels, DType* composite) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int s = index % spatial_dim;
	const int n = index / spatial_dim;

    const DType* a = data1 + n * channels * spatial_dim + s;
    const DType* b = data2 + n * channels * spatial_dim + s;
    DType* c = composite + n * channels * spatial_dim + s;

    c[0 * spatial_dim] += a[0 * spatial_dim] + b[0 * spatial_dim] + a[0 * spatial_dim] * b[0 * spatial_dim] + a[3 * spatial_dim] * b[1 * spatial_dim];
    c[1 * spatial_dim] += a[1 * spatial_dim] + b[1 * spatial_dim] + a[1 * spatial_dim] * b[0 * spatial_dim] + a[4 * spatial_dim] * b[1 * spatial_dim];
    c[2 * spatial_dim] += a[2 * spatial_dim] + b[2 * spatial_dim] + a[2 * spatial_dim] * b[0 * spatial_dim] + a[5 * spatial_dim] * b[1 * spatial_dim];
    c[3 * spatial_dim] += a[3 * spatial_dim] + b[3 * spatial_dim] + a[0 * spatial_dim] * b[3 * spatial_dim] + a[3 * spatial_dim] * b[4 * spatial_dim];
    c[4 * spatial_dim] += a[4 * spatial_dim] + b[4 * spatial_dim] + a[1 * spatial_dim] * b[3 * spatial_dim] + a[4 * spatial_dim] * b[4 * spatial_dim];
    c[5 * spatial_dim] += a[5 * spatial_dim] + b[5 * spatial_dim] + a[2 * spatial_dim] * b[3 * spatial_dim] + a[5 * spatial_dim] * b[4 * spatial_dim];
  }
}

template <typename DType>
__global__ void AffineGeneratorForward(const int n, const DType* data, 
                                      const int spatial_dim, const int channels,
                                      const int kernel_size, DType* offset) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int s = index % spatial_dim;
	const int n = index / spatial_dim;

    const int offset_dim = 2 * kernel_size * kernel_size;
    const DType* a = data + n * channels * spatial_dim + s;
    DType* cur_offset = offset + n * offset_dim * spatial_dim + s;

    const int kernel_radius = kernel_size / 2;
    for (int dh = -kernel_radius; dh <= kernel_radius; dh++) {
        for (int dw = -kernel_radius; dw <= kernel_radius; dw++) {
            int h = dh + kernel_radius;
            int w = dw + kernel_radius;
            cur_offset[((h * kernel_size + w) * 2 + 0) * spatial_dim] += a[0 * spatial_dim] * dh + a[1 * spatial_dim] * dw + a[2 * spatial_dim];
            cur_offset[((h * kernel_size + w) * 2 + 1) * spatial_dim] += a[3 * spatial_dim] * dh + a[4 * spatial_dim] * dw + a[5 * spatial_dim];
        }
    }
  }
}

//[ a0 + b0 + a0*b0 + a3*b1, a1 + b1 + a1*b0 + a4*b1, a2 + b2 + a2*b0 + a5*b1 ]
//[ a3 + b3 + a0*b3 + a3*b4, a4 + b4 + a1*b3 + a4*b4, a5 + b5 + a2*b3 + a5*b4 ]
template <typename DType>
__global__ void AffineCompositeBackward(const int n, DType* data1_diff, DType* data2_diff,
                                      const int spatial_dim, const int channels,
                                      const DType* data1, const DType* data2, const DType* composite_diff) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int s = index % spatial_dim;
	const int n = index / spatial_dim;

    DType* a_diff = data1_diff + n * channels * spatial_dim + s;
    DType* b_diff = data2_diff + n * channels * spatial_dim + s;
    
    const DType* a = data1 + n * channels * spatial_dim + s;
    const DType* b = data2 + n * channels * spatial_dim + s;
    const DType* c_diff = composite_diff + n * channels * spatial_dim + s;

    
    a_diff[0*spatial_dim] += c_diff[0*spatial_dim] + c_diff[0*spatial_dim] * b[0*spatial_dim] + c_diff[3*spatial_dim] * b[3*spatial_dim];
    a_diff[1*spatial_dim] += c_diff[1*spatial_dim] + c_diff[1*spatial_dim] * b[0*spatial_dim] + c_diff[4*spatial_dim] * b[3*spatial_dim];
    a_diff[2*spatial_dim] += c_diff[2*spatial_dim] + c_diff[2*spatial_dim] * b[0*spatial_dim] + c_diff[5*spatial_dim] * b[3*spatial_dim];
    a_diff[3*spatial_dim] += c_diff[3*spatial_dim] + c_diff[0*spatial_dim] * b[1*spatial_dim] + c_diff[3*spatial_dim] * b[4*spatial_dim];
    a_diff[4*spatial_dim] += c_diff[4*spatial_dim] + c_diff[1*spatial_dim] * b[1*spatial_dim] + c_diff[4*spatial_dim] * b[4*spatial_dim];
    a_diff[5*spatial_dim] += c_diff[5*spatial_dim] + c_diff[2*spatial_dim] * b[1*spatial_dim] + c_diff[5*spatial_dim] * b[4*spatial_dim];
                          
    b_diff[0*spatial_dim] += c_diff[0*spatial_dim]
                           + c_diff[0*spatial_dim] * a[0*spatial_dim] + c_diff[1*spatial_dim] * a[1*spatial_dim] + c_diff[2*spatial_dim] * a[2*spatial_dim];
    b_diff[1*spatial_dim] += c_diff[1*spatial_dim]
                           + c_diff[0*spatial_dim] * a[3*spatial_dim] + c_diff[1*spatial_dim] * a[4*spatial_dim] + c_diff[2*spatial_dim] * a[5*spatial_dim];
    b_diff[2*spatial_dim] += c_diff[2*spatial_dim];
    b_diff[3*spatial_dim] += c_diff[3*spatial_dim]
                           + c_diff[3*spatial_dim] * a[0*spatial_dim] + c_diff[4*spatial_dim] * a[1*spatial_dim] + c_diff[5*spatial_dim] * a[2*spatial_dim];
    b_diff[4*spatial_dim] += c_diff[4*spatial_dim]
                           + c_diff[3*spatial_dim] * a[3*spatial_dim] + c_diff[4*spatial_dim] * a[4*spatial_dim] + c_diff[5*spatial_dim] * a[5*spatial_dim];
    b_diff[5*spatial_dim] += c_diff[5*spatial_dim];
  }
}

template <typename DType>
__global__ void AffineGeneratorBackward(const int n, DType* data_diff, 
                                      const int spatial_dim, const int channels,
                                      const int kernel_size, const DType* offset_diff) {
  CUDA_KERNEL_LOOP(index, n) { 
	const int s = index % spatial_dim;
	const int n = index / spatial_dim;

    const int offset_dim = 2 * kernel_size * kernel_size;
    DType* a_diff = data_diff + n * channels * spatial_dim + s;
    const DType* cur_offset_diff = offset_diff + n * offset_dim * spatial_dim + s;

    const int kernel_radius = kernel_size / 2;
    for (int dh = -kernel_radius; dh <= kernel_radius; dh++) {
        for (int dw = -kernel_radius; dw <= kernel_radius; dw++) {
            //cur_offset[((h * kernel_size + w) * 2 + 0) * spatial_dim] = a[0 * spatial_dim] * dh + a[1 * spatial_dim] * dw + a[2 * spatial_dim];
            //cur_offset[((h * kernel_size + w) * 2 + 1) * spatial_dim] = a[3 * spatial_dim] * dh + a[4 * spatial_dim] * dw + a[5 * spatial_dim];
            int h = dh + kernel_radius;
            int w = dw + kernel_radius;
            DType cur_offset_diff_h = cur_offset_diff[((h * kernel_size + w) * 2 + 0) * spatial_dim];
            DType cur_offset_diff_w = cur_offset_diff[((h * kernel_size + w) * 2 + 1) * spatial_dim];
            a_diff[0 * spatial_dim] += cur_offset_diff_h * dh;
            a_diff[1 * spatial_dim] += cur_offset_diff_h * dw;
            a_diff[2 * spatial_dim] += cur_offset_diff_h;
            a_diff[3 * spatial_dim] += cur_offset_diff_w * dh;
            a_diff[4 * spatial_dim] += cur_offset_diff_w * dw;
            a_diff[5 * spatial_dim] += cur_offset_diff_w;
        }
    }
  }
}

}  // namespace offset_generator
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class OffsetGeneratorGPUOp : public Operator{
 public:
  explicit OffsetGeneratorGPUOp(OffsetGeneratorParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda::offset_generator;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data;
    if (param_.transform_composite) {
        Tensor<xpu, 4, DType> data1 = in_data[offsetGenerator::kData1].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> data2 = in_data[offsetGenerator::kData2].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> composite = out_data[offsetGenerator::kComposite].get<xpu, 4, DType>(s);
        
        if (req[offsetGenerator::kComposite] == kWriteTo)
            composite = 0;
        
        index_t batch_num = data1.shape_[0];
        index_t channels = data1.shape_[1];
        index_t height = data1.shape_[2];
        index_t width = data1.shape_[3];
        
        index_t num_kernels = batch_num * height * width;        
        AffineCompositeForward // NOLINT_NEXT_LINE(whitespace/operators)
              <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
              (num_kernels, data1.dptr_, data2.dptr_, height * width, channels, composite.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(AffineCompositeForward);
        
        data = composite;
    } else {
        data = in_data[offsetGenerator::kData1].get<xpu, 4, DType>(s);
    }

    Tensor<xpu, 4, DType> offset = out_data[offsetGenerator::kOffset].get<xpu, 4, DType>(s);
    if (req[offsetGenerator::kOffset] == kWriteTo)
            offset = 0;
    
    index_t batch_num = data.shape_[0];
    index_t channels = data.shape_[1];
    index_t height = data.shape_[2];
    index_t width = data.shape_[3];
    
    index_t num_kernels = batch_num * height * width;
    AffineGeneratorForward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, data.dptr_, height * width, channels, param_.kernel_size, offset.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(AffineGeneratorForward);
    
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
    using namespace mshadow::cuda::offset_generator;
    using namespace mxnet_op;    
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data_diff;
    if (param_.transform_composite) {
        Tensor<xpu, 4, DType> composite_diff = out_grad[offsetGenerator::kComposite].get<xpu, 4, DType>(s);
        
        Tensor<xpu, 1, DType> workspace =
        ctx.requested[offsetGenerator::kTempResource].get_space_typed<xpu, 1, DType>(
            Shape1(composite_diff.shape_[0]*composite_diff.shape_[1]*composite_diff.shape_[2]*composite_diff.shape_[3]), s);
        data_diff = Tensor<xpu, 4, DType>(workspace.dptr_, composite_diff.shape_, s);
        data_diff = 0;

        data_diff += composite_diff;
    } else {
        data_diff = in_grad[offsetGenerator::kData1].get<xpu, 4, DType>(s);
        if (req[offsetGenerator::kData1] == kWriteTo)
            data_diff = 0;
    }
    
    Tensor<xpu, 4, DType> offset_diff = out_grad[offsetGenerator::kOffset].get<xpu, 4, DType>(s);
    
    index_t batch_num = data_diff.shape_[0];
    index_t channels = data_diff.shape_[1];
    index_t height = data_diff.shape_[2];
    index_t width = data_diff.shape_[3];
    
    index_t num_kernels = batch_num * height * width;
    AffineGeneratorBackward // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
          (num_kernels, data_diff.dptr_, height * width, channels, param_.kernel_size, offset_diff.dptr_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(AffineGeneratorBackward);
    
    if (param_.transform_composite) {
        Tensor<xpu, 4, DType> data1_diff = in_grad[offsetGenerator::kData1].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> data2_diff = in_grad[offsetGenerator::kData2].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> data1 = in_data[offsetGenerator::kData1].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> data2 = in_data[offsetGenerator::kData2].get<xpu, 4, DType>(s);
        
        if (req[offsetGenerator::kData1] == kWriteTo)
            data1_diff = 0;
        if (req[offsetGenerator::kData2] == kWriteTo)
            data2_diff = 0;
        
        index_t batch_num = data_diff.shape_[0];
        index_t channels = data_diff.shape_[1];
        index_t height = data_diff.shape_[2];
        index_t width = data_diff.shape_[3];
        
        index_t num_kernels = batch_num * height * width;        
        AffineCompositeBackward // NOLINT_NEXT_LINE(whitespace/operators)
              <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>
              (num_kernels, data1_diff.dptr_, data2_diff.dptr_, height * width, channels, 
              data1.dptr_, data2.dptr_, data_diff.dptr_);
        MSHADOW_CUDA_POST_KERNEL_CHECK(AffineCompositeBackward);
    }
  }

 private:
  OffsetGeneratorParam param_;
};  // class OffsetGeneratorGPUOp

template<>
Operator* CreateOp<gpu>(OffsetGeneratorParam param) {
  return new OffsetGeneratorGPUOp<gpu, real_t>(param);
}
}  // namespace op
}  // namespace mxnet
