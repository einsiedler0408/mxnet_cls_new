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
 * \file bilinear_sampler_v2-inl.h
 * \brief
 * \author Xu Dong, Yunsheng Tian, Han Hu
*/
#ifndef MXNET_OPERATOR_BILINEAR_SAMPLER_V2_INL_H_
#define MXNET_OPERATOR_BILINEAR_SAMPLER_V2_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace bs_v2 {
enum BilinearSamplerV2OpInputs {kData, kGrid};
enum BilinearSamplerV2OpOutputs {kOut, kTmp};
}

struct BilinearSamplerV2Param : public dmlc::Parameter<BilinearSamplerV2Param> {
  DMLC_DECLARE_PARAMETER(BilinearSamplerV2Param) {
  }
};

template<typename xpu, typename DType>
class BilinearSamplerV2Op : public Operator {
 public:
  explicit BilinearSamplerV2Op(BilinearSamplerV2Param p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[bs_v2::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[bs_v2::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grid = in_data[bs_v2::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[bs_v2::kOut].get<xpu, 4, DType>(s);

    BilinearSamplerV2Forward(out, data, grid);
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
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[bs_v2::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grid = in_data[bs_v2::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[bs_v2::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> ggrid = in_grad[bs_v2::kGrid].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad = out_grad[bs_v2::kOut].get<xpu, 4, DType>(s);
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
    } else if (req[bs_v2::kData] != kNullOp && req[bs_v2::kGrid] == kNullOp) {
      if (req[bs_v2::kData] == kWriteTo) {
        gdata = scalar<DType>(0.0f);
      }
      BilinearSamplerV2DataBackward(gdata, grad, data, grid);
    }
    else {
      LOG(FATAL) << "Have not implemented the data req combinations! gdata_req="
                 << req[bs_v2::kData] << " ggrid_req=" << req[bs_v2::kGrid];
    }
  }

 private:
  BilinearSamplerV2Param param_;
};  // class BilinearSamplerV2Op

template<typename xpu>
Operator* CreateOp(BilinearSamplerV2Param param, int dtype);

#if DMLC_USE_CXX11
class BilinearSamplerV2Prop : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "grid"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "tmp"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, grid]";
    const TShape &dshape = (*in_shape)[bs_v2::kData];
    const TShape &lshape = (*in_shape)[bs_v2::kGrid];
    if (dshape.ndim() == 0) return false;
    CHECK_EQ(dshape.ndim(), 4U) \
        << "input data should be 4D in batch-y-x-num_filter";
    if (lshape.ndim() ==  0) return false;
    CHECK_EQ(lshape.ndim(), 4U) \
      << "Sampler grid should be 4D in batch-2-y-x";
    CHECK_EQ(dshape[0], lshape[0]);
    CHECK_EQ(lshape[1], 2U) << "incorrect grid shape[1], should be 2";
    // target height
    CHECK_GT(lshape[2], 0U) \
            << "incorrect grid_shape: " << lshape[2];
    // target width
    CHECK_GT(lshape[3], 0U) \
        << "incorrect grid_shape: " << lshape[3];
    out_shape->clear();
    // output_shape : (data.shape[0], grid.shape[2], grid.shape[3], data.shape[3])
    out_shape->push_back(dshape);
    (*out_shape)[bs_v2::kOut][1] = lshape[2];
    (*out_shape)[bs_v2::kOut][2] = lshape[3];
    out_shape->push_back(Shape4(lshape[0], lshape[2], lshape[3], 2));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      int dtype = -1;
      for (size_t i = 0; i < in_type->size(); ++i) {
        if (dtype == -1) {
          dtype = in_type->at(i);
        } else {
          CHECK(in_type->at(i) == dtype ||
                in_type->at(i) == -1) <<
                "Non-uniform data type in BilinearSamplerV2";
        }
      }
      if (dtype == -1) {
        LOG(FATAL) << "Not enough information to infer type in BilinearSamplerV2.";
        return false;
      }
      size_t nin = this->ListArguments().size();
      in_type->clear();
      for (size_t i = 0; i < nin; ++i) in_type->push_back(dtype);
      size_t naux = this->ListAuxiliaryStates().size();
      aux_type->clear();
      for (size_t i = 0; i < naux; ++i) aux_type->push_back(dtype);
      size_t nout = this->ListOutputs().size();
      out_type->clear();
      for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
      return true;
    }

  OperatorProperty* Copy() const override {
    auto ptr = new BilinearSamplerV2Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BilinearSamplerV2";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[bs_v2::kOut],
            in_data[bs_v2::kData],
            out_data[bs_v2::kTmp],
            in_data[bs_v2::kGrid]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BilinearSamplerV2Param param_;
};  // class BilinearSamplerV2Prop
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BILINEAR_SAMPLER_V2_INL_H_
