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
 * Copyright (c) 2017 by Contributors
 * \file bilinear_sampler_v2.cu
 * \brief
 * \author Xu Dong, Yunsheng Tian, Han Hu
*/

#include "./bilinear_sampler_v2-inl.h"
#include <algorithm>
#include "../common/cuda_utils.h"
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_bilinear_sampler_v2-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

namespace mshadow {
namespace cuda {
template<typename DType>
__device__ bool between(DType value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}
template<typename DType>
__global__ void BilinearSamplerV2ForwardKernel(const int i_h, const int i_w,
                                              const int i_c, const DType* data,
                                              const DType* grid, const int o_n,
                                              const int o_h, const int o_w,
                                              const int o_c, DType* out) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < o_n * o_h * o_w * o_c;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, h, w, c) is the element in out
    int c = index % o_c;
    int w = (index / o_c) % o_w;
    int h = (index / o_c / o_w) % o_h;
    int n = index / o_c / o_w / o_h;
    index_t grid_index = n * o_h * o_w * 2 + h * o_w + w;

    DType y_real, x_real;
    y_real = (*(grid + grid_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    x_real = (*(grid + grid_index) + 1) * (i_w - 1) / 2;
    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);

    index_t out_index = n * o_h * o_w * o_c + h * o_w * o_c + w * o_c + c;
    int data_index = n * i_h * i_w * i_c + top_left_y * i_w * i_c +
      top_left_x * i_c + c;
    DType top_left_v = 0;
    DType top_right_v = 0;
    DType bottom_left_v = 0;
    DType bottom_right_v = 0;
    if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1))
      top_left_v = *(data + data_index);
    if (between(top_left_x + 1, 0, i_w-1) && between(top_left_y, 0, i_h-1))
      top_right_v = *(data + data_index + i_c);
    if (between(top_left_x, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
      bottom_left_v = *(data + data_index + i_w * i_c);
    if (between(top_left_x+1, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
      bottom_right_v = *(data + data_index + i_w * i_c + i_c);
    *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
                        top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                        bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                        bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);

  }
}

template<typename DType>
__global__ void BilinearSamplerV2DataBackwardKernel(const int i_h, const int i_w,
                                              const int i_c, const DType* grad,
                                              const DType* data, const int o_n,
                                              const int o_h, const int o_w,
                                              const int o_c, DType* g_input,
                                              const DType* grid_src,
                                              int num_block_span) {
  //extern __shared__ char sharemem[];
  int o_cc = blockDim.x * num_block_span;

  for (int index = num_block_span * ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x);
       index < o_n * o_h * o_w * o_cc;
       index += num_block_span * blockDim.x * gridDim.x * gridDim.y) {
    // (n, h, w, c) is the element in grad
    int c = index % o_cc;
    int w = (index / o_cc) % o_w;
    int h = (index / o_cc / o_w) % o_h;
    int n = index / o_cc / o_w / o_h;
    index_t grid_src_index = n * o_h * o_w * 2 + h * o_w + w;

    DType y_real, x_real;
    y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;
    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);

    for (int b = 0; b < num_block_span; ++b) {

      if (c + b < o_c) {
        index_t grad_index = n * o_h * o_w * o_c + h * o_w * o_c + w * o_c + c + b;
        int data_index = n * i_h * i_w * i_c + top_left_y * i_w * i_c + top_left_x * i_c + c + b;
        // calc 4 vertex value in input data
        DType top_left_v = 0;
        DType top_right_v = 0;
        DType bottom_left_v = 0;
        DType bottom_right_v = 0;
        // calc input grad
        DType grad_val = *(grad + grad_index);
        if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
          atomicAdd(&g_input[data_index], grad_val * top_left_y_w * top_left_x_w);
          top_left_v = *(data + data_index);
        }
        if (between(top_left_x+1, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
          atomicAdd(&g_input[data_index + i_c], grad_val * top_left_y_w
                                          * (1.0 - top_left_x_w));
          top_right_v = *(data + data_index + i_c);
        }
        if (between(top_left_x, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
          atomicAdd(&g_input[data_index + i_w * i_c], grad_val * (1.0 - top_left_y_w)
                                          * top_left_x_w);
          bottom_left_v = *(data + data_index + i_w * i_c);
        }
        if (between(top_left_x+1, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
          atomicAdd(&g_input[data_index + i_w * i_c + i_c], grad_val * (1.0 - top_left_y_w)
                                              * (1.0 - top_left_x_w));
          bottom_right_v = *(data + data_index + i_w * i_c + i_c);
        }
      }

    }
  }
}

template<typename DType>
__global__ void BilinearSamplerV2BackwardKernel(const int i_h, const int i_w,
                                              const int i_c, const DType* grad,
                                              const DType* data, const int o_n,
                                              const int o_h, const int o_w,
                                              const int o_c, DType* g_input,
                                              const DType* grid_src,
                                              DType* grad_grid, int num_block_span) {
  extern __shared__ char sharemem[];
  int o_cc = blockDim.x * num_block_span;

  for (int index = num_block_span * ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x);
       index < o_n * o_h * o_w * o_cc;
       index += num_block_span * blockDim.x * gridDim.x * gridDim.y) {
    // (n, h, w, c) is the element in grad
    int c = index % o_cc;
    int w = (index / o_cc) % o_w;
    int h = (index / o_cc / o_w) % o_h;
    int n = index / o_cc / o_w / o_h;
    index_t grid_src_index = n * o_h * o_w * 2 + h * o_w + w;

    DType y_real, x_real;
    y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;
    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);

    index_t top_left_y_gw_id = threadIdx.x * sizeof(DType) / sizeof(char);
    index_t top_left_x_gw_id = (blockDim.x + threadIdx.x) * sizeof(DType) / sizeof(char);
    DType* top_left_y_gw = (DType*)&sharemem[top_left_y_gw_id];
    DType* top_left_x_gw = (DType*)&sharemem[top_left_x_gw_id];

    index_t top_left_y_gw_sum_id = 2 * blockDim.x * sizeof(DType) / sizeof(char);
    index_t top_left_x_gw_sum_id = top_left_y_gw_sum_id + sizeof(DType) / sizeof(char);
    DType* top_left_y_gw_sum = (DType*)&sharemem[top_left_y_gw_sum_id];
    DType* top_left_x_gw_sum = (DType*)&sharemem[top_left_x_gw_sum_id];
    *top_left_y_gw_sum = 0.;
    *top_left_x_gw_sum = 0.;

    for (int b = 0; b < num_block_span; ++b) {

      if (c + b < o_c) {
        index_t grad_index = n * o_h * o_w * o_c + h * o_w * o_c + w * o_c + c + b;
        int data_index = n * i_h * i_w * i_c + top_left_y * i_w * i_c + top_left_x * i_c + c + b;
        // calc 4 vertex value in input data
        DType top_left_v = 0;
        DType top_right_v = 0;
        DType bottom_left_v = 0;
        DType bottom_right_v = 0;
        // calc input grad
        DType grad_val = *(grad + grad_index);
        if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
          atomicAdd(&g_input[data_index], grad_val * top_left_y_w * top_left_x_w);
          top_left_v = *(data + data_index);
        }
        if (between(top_left_x+1, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
          atomicAdd(&g_input[data_index + i_c], grad_val * top_left_y_w
                                          * (1.0 - top_left_x_w));
          top_right_v = *(data + data_index + i_c);
        }
        if (between(top_left_x, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
          atomicAdd(&g_input[data_index + i_w * i_c], grad_val * (1.0 - top_left_y_w)
                                          * top_left_x_w);
          bottom_left_v = *(data + data_index + i_w * i_c);
        }
        if (between(top_left_x+1, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
          atomicAdd(&g_input[data_index + i_w * i_c + i_c], grad_val * (1.0 - top_left_y_w)
                                              * (1.0 - top_left_x_w));
          bottom_right_v = *(data + data_index + i_w * i_c + i_c);
        }
        // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
        *top_left_y_gw = -grad_val * (top_right_v - bottom_right_v +
                          (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                          * top_left_x_w);
        *top_left_x_gw = -grad_val * (bottom_left_v - bottom_right_v +
                          (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                          * top_left_y_w);
      }

      // calc grad of grid
      for (int i = o_cc; i > num_block_span; i >>= 1) {
        __syncthreads();
        if (i % (2 * num_block_span) != 0) {
          i -= num_block_span;
          if (c == 0 && i < o_c) {
            *top_left_y_gw += *(top_left_y_gw + i / num_block_span);
            *top_left_x_gw += *(top_left_x_gw + i / num_block_span);
          }
        }
        if (c < (i >> 1) && c + b < o_c && c + b + (i >> 1) < o_c) {
          *top_left_y_gw += *(top_left_y_gw + (i >> 1) / num_block_span);
          *top_left_x_gw += *(top_left_x_gw + (i >> 1) / num_block_span);
        }
      }
      if (c == 0) {
        *top_left_y_gw_sum += *top_left_y_gw;
        *top_left_x_gw_sum += *top_left_x_gw;
      }
    }
    if (c == 0) {
      *(grad_grid + grid_src_index + o_h * o_w) += *top_left_y_gw_sum * (i_h - 1) / 2;
      *(grad_grid + grid_src_index) += *top_left_x_gw_sum * (i_w - 1) / 2;
    }
  }
}
}  // namespace cuda

template<typename DType>
inline DType clamp(DType value, DType lowerBound, DType upperBound) {
  return max(lowerBound, min(value, upperBound));
}

template<typename DType>
inline void BilinearSamplerV2Forward(const Tensor<gpu, 4, DType> &output,
                                    const Tensor<gpu, 4, DType> &input,
                                    const Tensor<gpu, 4, DType> &grid_src) {
    DType *out = output.dptr_;
    const DType *data = input.dptr_;
    const DType *grid = grid_src.dptr_;
    int o_n = output.size(0), o_h = output.size(1), o_w = output.size(2), o_c = output.size(3);
    int i_h = input.size(1), i_w = input.size(2), i_c = input.size(3);
    using namespace cuda;
    int num_thread = clamp(o_c / 4, 128, kMaxThreadsPerBlock);  // optimal thread num
    const int max_block = (o_n * o_h * o_w * o_c + num_thread - 1) / num_thread;
    const int grid_dim_x = (max_block > kMaxGridDim) ? kMaxGridDim : max_block;
    const int grid_dim_y =
      (max_block > kMaxGridDim) ? (max_block + kMaxGridDim - 1) / kMaxGridDim : 1;
    dim3 num_blocks(grid_dim_x, grid_dim_y);
    dim3 threads_per_block(num_thread);
    CheckLaunchParam(num_blocks, threads_per_block, "bilinear sampler forward");
    cudaStream_t stream = Stream<gpu>::GetStream(output.stream_);
    cuda::BilinearSamplerV2ForwardKernel<DType> << <num_blocks, threads_per_block, 0, stream>> >(
      i_h, i_w, i_c, data, grid, o_n, o_h, o_w, o_c, out);
    // post kernel check
    cudaError err = cudaPeekAtLastError();
    CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template<typename DType>
inline void BilinearSamplerV2DataBackward(const Tensor<gpu, 4, DType> &input_grad,
                                     const Tensor<gpu, 4, DType> &output_grad,
                                     const Tensor<gpu, 4, DType> &input_data,
                                     const Tensor<gpu, 4, DType> &grid) {
  DType *g_input = input_grad.dptr_;
  const DType *grid_src = grid.dptr_;
  const DType *grad = output_grad.dptr_;
  const DType *data = input_data.dptr_;
  int o_n = output_grad.size(0), o_h = output_grad.size(1),
      o_w = output_grad.size(2), o_c = output_grad.size(3);
  int i_h = input_data.size(1), i_w = input_data.size(2), i_c = input_data.size(3);
  using namespace cuda;
  int num_thread, num_block_span;
  if ((o_c & (o_c - 1)) == 0) { // power of 2
    num_thread = clamp(o_c / 4, 128, kMaxThreadsPerBlock);  // optimal thread num
    num_block_span = int(ceilf(float(o_c) / num_thread));
  }
  else if (o_c < 4 * 128) {
    num_thread = int(ceilf(o_c / 32.)) * 32;
    num_block_span = 1;
  }
  else if (o_c < 4 * kMaxThreadsPerBlock) {
    num_thread = int(ceilf(o_c / 32.) * 32) / 4;
    num_block_span = 4;
  }
  else {
    num_thread = kMaxThreadsPerBlock;
    num_block_span = int(ceilf(float(o_c) / kMaxThreadsPerBlock));
  }
  int o_cc = num_thread * num_block_span;
  const int max_block = (o_n * o_h * o_w * o_cc + num_thread - 1) / num_thread;
  const int grid_dim_x = (max_block > kMaxGridDim) ? kMaxGridDim : max_block;
  const int grid_dim_y =
    (max_block > kMaxGridDim) ? (max_block + kMaxGridDim - 1) / kMaxGridDim : 1;
  dim3 num_blocks(grid_dim_x, grid_dim_y);
  dim3 threads_per_block(num_thread);
  CheckLaunchParam(num_blocks, threads_per_block, "bilinear sampler backward");
  cudaStream_t stream = Stream<gpu>::GetStream(input_grad.stream_);
  cuda::BilinearSamplerV2DataBackwardKernel<DType> << <num_blocks, threads_per_block, 
    (num_thread + 1) * 2 * sizeof(DType), stream >> >(
    i_h, i_w, i_c, grad, data, o_n, o_h, o_w, o_c, g_input, grid_src, num_block_span);
  // post kernel check
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template<typename DType>
inline void BilinearSamplerV2Backward(const Tensor<gpu, 4, DType> &input_grad,
                                     const Tensor<gpu, 4, DType> &ggrid,
                                     const Tensor<gpu, 4, DType> &output_grad,
                                     const Tensor<gpu, 4, DType> &input_data,
                                     const Tensor<gpu, 4, DType> &grid) {
  DType *g_input = input_grad.dptr_;
  DType *grad_grid = ggrid.dptr_;
  const DType *grid_src = grid.dptr_;
  const DType *grad = output_grad.dptr_;
  const DType *data = input_data.dptr_;
  int o_n = output_grad.size(0), o_h = output_grad.size(1),
      o_w = output_grad.size(2), o_c = output_grad.size(3);
  int i_h = input_data.size(1), i_w = input_data.size(2), i_c = input_data.size(3);
  using namespace cuda;
  int num_thread, num_block_span;
  if ((o_c & (o_c - 1)) == 0) { // power of 2
    num_thread = clamp(o_c / 4, 128, kMaxThreadsPerBlock);  // optimal thread num
    num_block_span = int(ceilf(float(o_c) / num_thread));
  }
  else if (o_c < 4 * 128) {
    num_thread = int(ceilf(o_c / 32.)) * 32;
    num_block_span = 1;
  }
  else if (o_c < 4 * kMaxThreadsPerBlock) {
    num_thread = int(ceilf(o_c / 32.) * 32) / 4;
    num_block_span = 4;
  }
  else {
    num_thread = kMaxThreadsPerBlock;
    num_block_span = int(ceilf(float(o_c) / kMaxThreadsPerBlock));
  }
  int o_cc = num_thread * num_block_span;
  const int max_block = (o_n * o_h * o_w * o_cc + num_thread - 1) / num_thread;
  const int grid_dim_x = (max_block > kMaxGridDim) ? kMaxGridDim : max_block;
  const int grid_dim_y =
    (max_block > kMaxGridDim) ? (max_block + kMaxGridDim - 1) / kMaxGridDim : 1;
  dim3 num_blocks(grid_dim_x, grid_dim_y);
  dim3 threads_per_block(num_thread);
  CheckLaunchParam(num_blocks, threads_per_block, "bilinear sampler backward");
  cudaStream_t stream = Stream<gpu>::GetStream(input_grad.stream_);
  cuda::BilinearSamplerV2BackwardKernel<DType> << <num_blocks, threads_per_block, 
    (num_thread + 1) * 2 * sizeof(DType), stream >> >(
    i_h, i_w, i_c, grad, data, o_n, o_h, o_w, o_c, g_input, grid_src, grad_grid, num_block_span);
  // post kernel check
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BilinearSamplerV2Param param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNBilinearSamplerV2Op<DType>(param);
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearSamplerV2Op<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
  return op;
}

}  // namespace op
}  // namespace mxnet
