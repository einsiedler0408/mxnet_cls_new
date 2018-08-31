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
 * Copyright (c) 2016 by Contributors
 * \file cudnn_bilinear_sampler_v2-inl.h
 * \brief
 * \author Xu Dong, Yunsheng Tian, Han Hu
*/
#ifndef MXNET_OPERATOR_CUDNN_BILINEAR_SAMPLER_V2_INL_H_
#define MXNET_OPERATOR_CUDNN_BILINEAR_SAMPLER_V2_INL_H_

#include <algorithm>
#include <vector>
#include "./bilinear_sampler_v2-inl.h"
namespace mxnet {
namespace op {
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
template<typename DType>
class CuDNNBilinearSamplerV2Op : public Operator {
 public:
  explicit CuDNNBilinearSamplerV2Op(BilinearSamplerV2Param param) {
    this->param_ = param;
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    sampler_ = CUDNN_SAMPLER_BILINEAR;
  }

  ~CuDNNBilinearSamplerV2Op() {
    if (init_cudnn_) {
      CUDNN_CALL(cudnnDestroySpatialTransformerDescriptor(st_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(req[bs_v2::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 2U);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    Tensor<gpu, 4, DType> data = in_data[bs_v2::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grid = in_data[bs_v2::kGrid].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grid_tmp = out_data[bs_v2::kTmp].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> out = out_data[bs_v2::kOut].get<gpu, 4, DType>(s);
    // grid_tmp : (batch, h, w, 2)
    grid_tmp = transpose(grid, Shape4(0, 2, 3, 1));
    if (!init_cudnn_) {
     Init(s, in_data, out_data);
    }
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(grid_tmp.CheckContiguous(), true);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    CUDNN_CALL(cudnnSpatialTfSamplerForward(s->dnn_handle_,
                                            st_desc_,
                                            &alpha,
                                            in_desc_,
                                            data.dptr_,
                                            grid_tmp.dptr_,
                                            &beta,
                                            out_desc_,
                                            out.dptr_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_NE(req[bs_v2::kData], kWriteInplace);
    CHECK_NE(req[bs_v2::kGrid], kWriteInplace);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    Tensor<gpu, 4, DType> data = in_data[bs_v2::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grid = in_data[bs_v2::kGrid].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> gdata = in_grad[bs_v2::kData].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> ggrid = in_grad[bs_v2::kGrid].get<gpu, 4, DType>(s);
    Tensor<gpu, 4, DType> grad = out_grad[bs_v2::kOut].get<gpu, 4, DType>(s);
    if (req[bs_v2::kData] != kNullOp && req[bs_v2::kGrid] != kNullOp) {
      if (req[bs_v2::kData] == kWriteTo) {
        gdata = scalar<DType>(0.0f);
      }
      if (req[bs_v2::kGrid] == kWriteTo) {
        ggrid = scalar<DType>(0.0f);
      }
      BilinearSamplerV2Backward(gdata, ggrid, grad, data, grid);
    } else if (req[bs_v2::kData] == kNullOp && req[bs_v2::kGrid] == kNullOp) {
      return;
    } else {
      LOG(FATAL) << "Have not implemented the data req combinations! gdata_req="
                 << req[bs_v2::kData] << " ggrid_req=" << req[bs_v2::kGrid];
    }
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    #if CUDNN_MAJOR >= 5
    format_ = CUDNN_TENSOR_NHWC;
    #endif
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 2U);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      Tensor<gpu, 4, DType> data = in_data[bs_v2::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[bs_v2::kOut].get<gpu, 4, DType>(s);
      CUDNN_CALL(cudnnCreateSpatialTransformerDescriptor(&st_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                                            format_,
                                            dtype_,
                                            data.size(0),
                                            data.size(3),
                                            data.size(1),
                                            data.size(2)));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                            format_,
                                            dtype_,
                                            out.size(0),
                                            out.size(3),
                                            out.size(1),
                                            out.size(2)));
      int dim[] = {static_cast<int>(out.size(0)), static_cast<int>(out.size(3)),
                   static_cast<int>(out.size(1)), static_cast<int>(out.size(2))};
      CUDNN_CALL(cudnnSetSpatialTransformerNdDescriptor(st_desc_,
                                                        sampler_,
                                                        dtype_,
                                                        4,
                                                        dim));
    }
  }

  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnSpatialTransformerDescriptor_t st_desc_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnSamplerType_t sampler_;
  #if CUDNN_MAJOR >= 5
  cudnnTensorFormat_t format_;
  #endif
  BilinearSamplerV2Param param_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_BILINEAR_SAMPLER_V2_INL_H_
