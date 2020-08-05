/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PropagateConcatenationQparamPassInternal.h"

#include <luci/IR/CircleQuantParam.h>

#include <vector>

#include <gtest/gtest.h>

namespace
{

void addQuantParam(luci::CircleNode &node, const std::vector<float> &scale,
                   const std::vector<int64_t> &zp, int32_t quantized_dimension)
{
  assert(node.quantparam() == nullptr);

  auto quantparam = std::make_unique<luci::CircleQuantParam>();
  quantparam->scale = scale;
  quantparam->zerop = zp;
  quantparam->quantized_dimension = quantized_dimension;
  node.quantparam(std::move(quantparam));
}

class SimpleConcatGraph
{
public:
  SimpleConcatGraph()
  {
    concat_node.dtype(loco::DataType::U8);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(loco::DataType::U8);
    input_2.dtype(loco::DataType::U8);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    addQuantParam(concat_node, {3.14}, {77}, 0);
    addQuantParam(input_1, {1.0}, {1}, 0);
    addQuantParam(input_2, {2.0}, {2}, 0);
  }

  ~SimpleConcatGraph()
  {
    concat_node.values(0, nullptr);
    concat_node.values(1, nullptr);
  }

public:
  luci::CircleConcatenation concat_node{2};
  luci::CircleConv2D input_1;
  luci::CircleConv2D input_2;
};

class SubsequentConcatGraph
{
public:
  SubsequentConcatGraph()
  {
    concat_node.dtype(loco::DataType::U8);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(loco::DataType::U8);
    input_2.dtype(loco::DataType::U8);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    addQuantParam(concat_node, {3.14}, {77}, 0);
    addQuantParam(input_1, {1.0}, {1}, 0);
    addQuantParam(input_2, {2.0}, {2}, 0);
  }

  ~SubsequentConcatGraph()
  {
    concat_node.values(0, nullptr);
    concat_node.values(1, nullptr);
  }

public:
  luci::CircleConcatenation concat_node{2};
  luci::CircleConcatenation input_1{2};
  luci::CircleConv2D input_2;
};

} // namespace

TEST(PropagateConcatenationQparamPass, propagate_concat_quantparam)
{
  SimpleConcatGraph g;

  luci::propagate_concat_quantparam(&g.concat_node);

  EXPECT_FLOAT_EQ(g.concat_node.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.concat_node.quantparam()->zerop[0], 77);

  EXPECT_FLOAT_EQ(g.input_1.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.input_1.quantparam()->zerop[0], 77);

  EXPECT_FLOAT_EQ(g.input_2.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.input_2.quantparam()->zerop[0], 77);
}

TEST(PropagateConcatenationQparamPass, satisfy_concat_propagate_conditions)
{
  SimpleConcatGraph g;

  EXPECT_TRUE(luci::satisfy_concat_propagate_conditions(&g.concat_node));
}

TEST(PropagateConcatenationQparamPass, satisfy_concat_propagate_conditions_NEG)
{
  SimpleConcatGraph g;

  EXPECT_TRUE(luci::satisfy_concat_propagate_conditions(&g.concat_node));

  // (1) concat is uint8-quantized
  // (2) concat has no fused activation function
  // (3) inputs are uint8-quantized
  // (4) there is no subsequent concats
  // (5) inputs are not produced to Ops other than concat

  // concat is not uint8-quantized
  g.concat_node.dtype(loco::DataType::S8);
  EXPECT_FALSE(luci::satisfy_concat_propagate_conditions(&g.concat_node));
  g.concat_node.dtype(loco::DataType::U8);

  // concat has fused activation function
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_FALSE(luci::satisfy_concat_propagate_conditions(&g.concat_node));
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);

  // input is not uint8-quantized
  g.input_1.dtype(loco::DataType::S8);
  EXPECT_FALSE(luci::satisfy_concat_propagate_conditions(&g.concat_node));
  g.input_1.dtype(loco::DataType::U8);

  // input is given as an input of another Op (Op other than concat_node)
  g.input_2.input(&g.input_1);
  EXPECT_FALSE(luci::satisfy_concat_propagate_conditions(&g.concat_node));

  // subsequent concats exist
  SubsequentConcatGraph sg;
  EXPECT_FALSE(luci::satisfy_concat_propagate_conditions(&sg.concat_node));
}
