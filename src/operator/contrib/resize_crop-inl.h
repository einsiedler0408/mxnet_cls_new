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
 * \file resize_crop-inl.h
 * \brief ResizeCrop Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_RESIZE_CROP_INL_H_
#define MXNET_OPERATOR_CONTRIB_RESIZE_CROP_INL_H_

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

namespace resize_crop {
enum ResizeCropOpInputs {kData};
enum ResizeCropOpOutputs {kOutput, kOffset, kMask, kRandomness};
enum ResizeCropForwardResource {kRandom, kTempResource};
}  // resize_crop

struct ResizeCropParam : public dmlc::Parameter<ResizeCropParam> {
  float min_scale;
  float max_scale;
  bool keep_aspect_ratio;
  bool output_randomness;
  DMLC_DECLARE_PARAMETER(ResizeCropParam) {
    DMLC_DECLARE_FIELD(min_scale).set_default(1.0f)
    .describe("ResizeCrop min_scale");
    DMLC_DECLARE_FIELD(max_scale).set_default(1.0f)
    .describe("ResizeCrop max_scale");
    DMLC_DECLARE_FIELD(keep_aspect_ratio).set_default(true)
    .describe("ResizeCrop keep_aspect_ratio");
    DMLC_DECLARE_FIELD(output_randomness).set_default(false)
    .describe("Add randomness to outputs");
  }
};

template<typename xpu>
Operator *CreateOp(ResizeCropParam param);

#if DMLC_USE_CXX11
class ResizeCropProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(resize_crop::kData);
    if (dshape.ndim() == 0) return false;
    
    Shape<4> output_shape = Shape4(dshape[0], dshape[1], dshape[2], dshape[3]);
    Shape<4> offset_shape = Shape4(dshape[0], 2, dshape[2], dshape[3]);
    Shape<4> mask_shape = Shape4(dshape[0], 1, dshape[2], dshape[3]);
    Shape<4> randomness_shape = Shape4(dshape[0], 4, 1, 1);
    out_shape->clear();
    out_shape->push_back(output_shape);
    out_shape->push_back(offset_shape);
    out_shape->push_back(mask_shape);
    out_shape->push_back(randomness_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ResizeCropProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_ResizeCrop";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kRandom, ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ };
  }

  int NumVisibleOutputs() const override {
    if (param_.output_randomness) {
      return 4;
    } else {
      return 3;
    }
  }

  int NumOutputs() const override {
    return 4;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "offset", "mask", "randomness"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ResizeCropParam param_;
};  // class ResizeCropProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_RESIZE_CROP_INL_H_