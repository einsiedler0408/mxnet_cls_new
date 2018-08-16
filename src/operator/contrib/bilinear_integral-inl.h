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
 * \file bilinear_integral-inl.h
 * \brief BilinearIntegral Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_BILINEAR_INTEGRAL_INL_H_
#define MXNET_OPERATOR_CONTRIB_BILINEAR_INTEGRAL_INL_H_

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

namespace bilinear_integral {
enum BilinearIntegralOpInputs {kData, kOffset};
enum BilinearIntegralOpOutputs {kOutput};
enum BilinearIntegralForwardResource {kTempResource};
}  // bilinear_integral

struct BilinearIntegralParam : public dmlc::Parameter<BilinearIntegralParam> {
  DMLC_DECLARE_PARAMETER(BilinearIntegralParam) {
  }
};

template<typename xpu>
Operator *CreateOp(BilinearIntegralParam param);

#if DMLC_USE_CXX11
class BilinearIntegralProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, offset]";
    const TShape &dshape = in_shape->at(bilinear_integral::kData);
    if (dshape.ndim() == 0) return false;
    
    const TShape &oshape = in_shape->at(bilinear_integral::kOffset);
    CHECK_EQ(oshape[0], dshape[0]);
    CHECK_EQ(oshape[1], 2);
    
    Shape<4> output_shape = Shape4(dshape[0], dshape[1], oshape[2], oshape[3]);
    out_shape->clear();
    out_shape->push_back(output_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BilinearIntegralProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_BilinearIntegral";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[bilinear_integral::kOutput], in_data[bilinear_integral::kData], in_data[bilinear_integral::kOffset] };
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "offset"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  BilinearIntegralParam param_;
};  // class BilinearIntegralProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_BILINEAR_INTEGRAL_INL_H_