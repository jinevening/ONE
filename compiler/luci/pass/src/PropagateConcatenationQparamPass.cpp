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

#include "luci/Pass/PropagateConcatenationQparamPass.h"

#include "PropagateConcatenationQparamPassInternal.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>
#include <luci/Log.h>

namespace
{

bool is_uint8_quantized(luci::CircleNode *node)
{
  if (node->dtype() == loco::DataType::U8 && node->quantparam() != nullptr &&
      node->quantparam()->scale.size() > 0 && node->quantparam()->zerop.size() > 0)
    return true;

  return false;
}

void overwrite_quantparam(luci::CircleConcatenation *concat, luci::CircleNode *target)
{
  auto concat_qparam = concat->quantparam();
  assert(concat_qparam != nullptr);

  auto target_qparam = target->quantparam();
  assert(target_qparam != nullptr);
  target_qparam->min = concat_qparam->min;
  target_qparam->max = concat_qparam->max;
  target_qparam->scale = concat_qparam->scale;
  target_qparam->zerop = concat_qparam->zerop;
  target_qparam->quantized_dimension = concat_qparam->quantized_dimension;
}

} // namespace

namespace luci
{

bool satisfy_concat_propagate_conditions(luci::CircleConcatenation *concat)
{
  // Check if concat is uint8-quantized
  if (!is_uint8_quantized(concat))
    return false;

  // Check if concat has no fused activation function
  if (concat->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  const auto num_inputs = concat->numValues();
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto node = static_cast<luci::CircleNode *>(concat->arg(i));

    // Check if this input is quantized
    if (!is_uint8_quantized(node))
      return false;

    // Check if concat and node have the same quantized_dimension
    if (node->quantparam()->quantized_dimension != concat->quantparam()->quantized_dimension)
      return false;

    // Check if this input is not CONCAT Op
    if (node->opcode() == luci::CircleOpcode::CONCATENATION)
      return false;

    // Check if this input is not used by other Ops
    auto succs = loco::succs(node);
    if (succs.size() != 1)
      return false;

    assert(succs.find(concat) != succs.end());
  }

  return true;
}

void propagate_concat_quantparam(luci::CircleConcatenation *concat)
{
  const auto num_inputs = concat->numValues();
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto node = static_cast<luci::CircleNode *>(concat->arg(i));
    overwrite_quantparam(concat, node);
  }
}

/** BEFORE
 *
 *         [CircleNode]             [CircleNode]
 *           (qparam1)                (qparam2)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                          (qparam3)
 *
 *  AFTER
 *         [CircleNode]             [CircleNode]
 *           (qparam3)                (qparam3)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                          (qparam3)
 */
bool PropagateConcatenationQparamPass::run(loco::Graph *g)
{
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto concat = dynamic_cast<luci::CircleConcatenation *>(node);
    if (not concat)
      continue;

    // Check if
    // (1) concat is uint8-quantized
    // (2) concat has no fused activation function
    // (3) inputs are uint8-quantized
    // (4) there is no subsequent concats
    // (5) inputs are not produced to Ops other than concat
    if (satisfy_concat_propagate_conditions(concat))
      propagate_concat_quantparam(concat);
  }

  return false;
}

} // namespace luci
