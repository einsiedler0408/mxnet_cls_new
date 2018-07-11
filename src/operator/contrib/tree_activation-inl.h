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
 * \file tree_activation-inl.h
 * \brief TreeActivation Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu
*/
#ifndef MXNET_OPERATOR_CONTRIB_TREE_ACTIVATION_INL_H_
#define MXNET_OPERATOR_CONTRIB_TREE_ACTIVATION_INL_H_

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

namespace treeAct {
enum TreeActivationOpInputs {kData};
enum TreeActivationOpOutputs {kValue, kIndex};
enum TreeActivationForwardResource {kTempResource};
}  // treeAct

struct TreeActivationParam : public dmlc::Parameter<TreeActivationParam> {
  int tree_depth;
  int tree_num;
  
  DMLC_DECLARE_PARAMETER(TreeActivationParam) {
    DMLC_DECLARE_FIELD(tree_depth).set_default(1).describe("tree_depth");
    DMLC_DECLARE_FIELD(tree_num).set_default(1).describe("tree_num");
  }
};

template<typename xpu>
Operator *CreateOp(TreeActivationParam param);

#if DMLC_USE_CXX11
class TreeActivationProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(treeAct::kData);
    if (dshape.ndim() == 0) return false;
    
    int node_num = (1 << param_.tree_depth) - 1;
    CHECK_EQ(dshape[1], node_num * param_.tree_num);
    
    Shape<4> output_shape = Shape4(dshape[0], param_.tree_depth * param_.tree_num, dshape[2], dshape[3]);

    out_shape->clear();
    // value
    out_shape->push_back(output_shape);
    // index
    out_shape->push_back(output_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new TreeActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_TreeActivation";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[treeAct::kValue], out_data[treeAct::kIndex]};
  }

  int NumVisibleOutputs() const override {
    return 2;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"value", "index"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  TreeActivationParam param_;
};  // class TreeActivationProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_TREE_ACTIVATION_INL_H_