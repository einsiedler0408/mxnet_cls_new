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
 * \file offset_generator-inl.h
 * \brief OffsetGenerator Operator
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

namespace offsetGenerator {
enum OffsetGeneratorOpInputs {kData1, kData2};
enum OffsetGeneratorOpOutputs {kOffset, kComposite};
enum OffsetGeneratorForwardResource {kTempResource};
enum OffsetGeneratorTransformType {kAffine};
}  // offsetGenerator

struct OffsetGeneratorParam : public dmlc::Parameter<OffsetGeneratorParam> {
  int  kernel_size;
  bool transform_composite;
  int  transform_type;
    
  DMLC_DECLARE_PARAMETER(OffsetGeneratorParam) {
    DMLC_DECLARE_FIELD(kernel_size).set_default(3).describe("kernel = (kernel_size, kernel_size)");
    DMLC_DECLARE_FIELD(transform_composite).set_default(false).describe("transform_composite");
    DMLC_DECLARE_FIELD(transform_type)
    .add_enum("affine", offsetGenerator::kAffine)
    .describe("The type of transformation. For `affine`, input data should be an affine matrix ");
  }
};

template<typename xpu>
Operator *CreateOp(OffsetGeneratorParam param);

#if DMLC_USE_CXX11
class OffsetGeneratorProp : public OperatorProperty {
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
    if (param_.transform_composite) {
        CHECK_EQ(in_shape->size(), 2) << "Input:[data1, data2]";
        const TShape &dshape = in_shape->at(offsetGenerator::kData1);
        if (dshape.ndim() == 0) return false;
 
        Shape<4> output_shape = Shape4(dshape[0], param_.kernel_size*param_.kernel_size*2, dshape[2], dshape[3]);
        out_shape->clear();
        out_shape->push_back(output_shape);
        out_shape->push_back(Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));
        
        return true;
    }
    else {
        CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
        const TShape &dshape = in_shape->at(offsetGenerator::kData1);
        if (dshape.ndim() == 0) return false;
 
        Shape<4> output_shape = Shape4(dshape[0], param_.kernel_size*param_.kernel_size*2, dshape[2], dshape[3]);
        out_shape->clear();
        out_shape->push_back(output_shape);

        return true;
    }
    
  }

  OperatorProperty* Copy() const override {
    auto ptr = new OffsetGeneratorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_OffsetGenerator";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {};
  }
  
  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.transform_composite)
        return {out_grad[offsetGenerator::kOffset], out_grad[offsetGenerator::kComposite],
                in_data[offsetGenerator::kData1], in_data[offsetGenerator::kData2]};
    else
        return {out_grad[offsetGenerator::kOffset]};
  }

  int NumVisibleOutputs() const override {
    if (param_.transform_composite)
        return 2;
    else
        return 1;
  }

  int NumOutputs() const override {
    if (param_.transform_composite)
        return 2;
    else
        return 1;
  }

  std::vector<std::string> ListArguments() const override {
    if (param_.transform_composite)
        return {"data1", "data2"};
    else
        return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    if (param_.transform_composite)
        return {"offset", "composite"};
    else
        return {"offset"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  OffsetGeneratorParam param_;
};  // class OffsetGeneratorProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_OFFSET_GENERATOR_INL_H_