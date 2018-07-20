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
 * \file bilinear_downsample-inl.h
 * \brief BilinearDownsample Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_BILINEAR_DOWNSAMPLE_INL_H_
#define MXNET_OPERATOR_CONTRIB_BILINEAR_DOWNSAMPLE_INL_H_

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

namespace bilinear_downsample {
enum BilinearDownsampleOpInputs {kData};
enum BilinearDownsampleOpOutputs {kOutput};
enum BilinearDownsampleForwardResource {kTempResource};
}  // bilinear_downsample

struct BilinearDownsampleParam : public dmlc::Parameter<BilinearDownsampleParam> {
  float rescale;
  DMLC_DECLARE_PARAMETER(BilinearDownsampleParam) {
    DMLC_DECLARE_FIELD(rescale).set_default(2.0f).set_range(1.0f, 10000.0f)
    .describe("BilinearDownsample scale");
  }
};

template<typename xpu>
Operator *CreateOp(BilinearDownsampleParam param);

#if DMLC_USE_CXX11
class BilinearDownsampleProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(bilinear_downsample::kData);
    if (dshape.ndim() == 0) return false;
    
    Shape<4> output_shape = Shape4(dshape[0], dshape[1], int((dshape[2]-1)/param_.rescale)+1, int((dshape[3]-1)/param_.rescale)+1);
    out_shape->clear();
    out_shape->push_back(output_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BilinearDownsampleProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_BilinearDownsample";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[bilinear_downsample::kOutput] };
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  BilinearDownsampleParam param_;
};  // class BilinearDownsampleProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_BILINEAR_DOWNSAMPLE_INL_H_