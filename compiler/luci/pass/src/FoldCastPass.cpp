/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FoldCastPass.h"

#include <luci/IR/CircleNodes.h>

#include <type_traits>

namespace
{

template <typename FromT, typename ToT> ToT safe_down_cast(FromT from)
{
  if (from < std::numeric_limits<ToT>::lowest())
    return std::numeric_limits<ToT>::lowest();

  if (from > std::numeric_limits<ToT>::max())
    return std::numeric_limits<ToT>::max();

  return static_cast<ToT>(from);
}

template <loco::DataType FromT, loco::DataType ToT>
void write_downcasted_const(const luci::CircleConst *from, luci::CircleConst *to)
{
  assert(FromT == from->dtype());
  assert(ToT == to->dtype());

  // Resize destination const
  const auto size = from->size<FromT>();
  to->size<ToT>(size);

  // TODO: Support more data types
  if (FromT == loco::DataType::S64)
  {
    if (ToT == loco::DataType::S32)
    {
      for (uint32_t i = 0; i < size; i++)
        to->at<ToT>(i) = safe_down_cast<int64_t, int32_t>(from->at<FromT>(i));
    }
  }
}

luci::CircleConst *cast_const(luci::CircleConst *node, loco::DataType from_dtype,
                              loco::DataType to_dtype)
{
  assert(node->dtype() == from_dtype);

  auto constant = node->graph()->nodes()->create<luci::CircleConst>();
  constant->dtype(to_dtype);
  constant->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    constant->dim(i).set(node->dim(i).value());

  constant->shape_status(luci::ShapeStatus::VALID);

  // TODO: Support more data types
  if (from_dtype == loco::DataType::S64)
  {
    if (to_dtype == loco::DataType::S32)
    {
      write_downcasted_const<loco::DataType::S64, loco::DataType::S32>(node, constant);
      return constant;
    }
    return nullptr;
  }

  return nullptr;
}

/**
 * Fold Cast to const if it has const input
 **/
bool fold_cast(luci::CircleCast *cast)
{
  // Check cast has const input
  auto const_x = dynamic_cast<luci::CircleConst *>(cast->x());
  if (not const_x)
    return false;

  const auto in_dtype = const_x->dtype();
  const auto out_dtype = cast->dtype();

  auto casted_const = cast_const(const_x, in_dtype, out_dtype);
  if (not casted_const)
    return false;

  loco::replace(cast).with(casted_const);

  return true;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for Cast Op
 **/
bool FoldCastPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto cast = dynamic_cast<luci::CircleCast *>(node))
    {
      if (fold_cast(cast))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
