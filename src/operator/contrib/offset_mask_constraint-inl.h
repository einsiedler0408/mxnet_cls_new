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
 * \file offset_mask_constraint-inl.h
 * \brief OffsetMaskConstraint Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_OFFSET_GENERATOR_INL_H_
#define MXNET_OPERATOR_CONTRIB_OFFSET_GENERATOR_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"


namespace mxnet {
namespace op {

namespace offsetMaskConstraint {
enum OffsetMaskConstraintOpInputs {kOffset, kMaskConstraint};
enum OffsetMaskConstraintOpOutputs {kOutput};
enum OffsetMaskConstraintForwardResource {kTempResource};
}  // offsetMaskConstraint

struct OffsetMaskConstraintParam : public dmlc::Parameter<OffsetMaskConstraintParam> {
  float grad_scale;
  bool border_constraint;
  int mask_offset_ratio;
  int conv_stride;
  int conv_dilate;
  int conv_kernel;
    
  DMLC_DECLARE_PARAMETER(OffsetMaskConstraintParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0).describe("grad_scale");
    DMLC_DECLARE_FIELD(border_constraint).set_default(true).describe("border_constraint");
    DMLC_DECLARE_FIELD(mask_offset_ratio).set_default(16).describe("mask_offset_ratio");
    DMLC_DECLARE_FIELD(conv_stride).set_default(1).describe("conv_stride");
    DMLC_DECLARE_FIELD(conv_dilate).set_default(1).describe("conv_dilate");
    DMLC_DECLARE_FIELD(conv_kernel).set_default(3).describe("conv_kernel");
  }
};

template<typename xpu>
Operator *CreateOp(OffsetMaskConstraintParam param);

#if DMLC_USE_CXX11
class OffsetMaskConstraintProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[offset, mask_constraint]";
    const TShape &dshape = in_shape->at(offsetMaskConstraint::kOffset);
    if (dshape.ndim() == 0) return false;
 
    out_shape->clear();
    out_shape->push_back(Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));
    return true;    
  }

  OperatorProperty* Copy() const override {
    auto ptr = new OffsetMaskConstraintProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_OffsetMaskConstraint";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {};
  }
  
  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[offsetMaskConstraint::kOutput], 
            in_data[offsetMaskConstraint::kOffset], in_data[offsetMaskConstraint::kMaskConstraint]};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"offset", "mask_constraint"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  OffsetMaskConstraintParam param_;
};  // class OffsetMaskConstraintProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_OFFSET_GENERATOR_INL_H_