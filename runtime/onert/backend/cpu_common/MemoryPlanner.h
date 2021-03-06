/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        MemoryPlanner.h
 * @brief       This file contains Memory Planning related classes
 */

#ifndef __ONERT_BACKEND_CPU_COMMON_MEMORY_PLANNER_H__
#define __ONERT_BACKEND_CPU_COMMON_MEMORY_PLANNER_H__

#include <map>
#include <unordered_set>
#include <memory>

#include "Allocator.h"
#include "ir/OperandIndexMap.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

/**
 * @brief Structure to have memory offset and size
 */
struct Block
{
  uint32_t offset;
  size_t size;
};

/**
 * @brief Interface to plan memory
 */
struct IMemoryPlanner
{
  using MemoryPlans = ir::OperandIndexMap<Block>;

  /**
   * @brief Claim memory for operand
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  virtual void claim(const ir::OperandIndex &, size_t) = 0;
  /**
   * @brief Release memory for operand
   * @param[in] index The operand index
   */
  virtual void release(const ir::OperandIndex &) = 0;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  virtual uint32_t capacity() = 0;
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  virtual MemoryPlans &memory_plans() = 0;

  virtual ~IMemoryPlanner() = default;
};

/**
 * @brief Class to plan memory by bump way
 */
class BumpPlanner : public IMemoryPlanner
{
public:
  /**
   * @brief Claim memory for operand by bump way
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  void claim(const ir::OperandIndex &, size_t) override;
  /**
   * @brief Release memory for operand by bump way
   * @param[in] index The operand index
   */
  void release(const ir::OperandIndex &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  MemoryPlans &memory_plans() override { return _mem_plans; }

private:
  uint32_t _capacity = 0;
  MemoryPlans _mem_plans;
};

/**
 * @brief Class to plan memory by firstfit way
 */
class FirstFitPlanner : public IMemoryPlanner
{
public:
  /**
   * @brief Claim memory for operand by firstfit way
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  void claim(const ir::OperandIndex &, size_t) override;
  /**
   * @brief Release memory for operand by firstfit way
   * @param[in] index The operand index
   */
  void release(const ir::OperandIndex &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  MemoryPlans &memory_plans() override { return _mem_plans; }

private:
  uint32_t _capacity = 0;
  MemoryPlans _mem_plans;
  // Use std::map because claim() assumes that _claim_table is sorted by uint32_t(base_offset)
  std::map<uint32_t, ir::OperandIndex> _claim_table;
};

/**
 * @brief Class to plan memory by Weighted Interval Color algorithm
 */
class WICPlanner : public IMemoryPlanner
{
public:
  WICPlanner();

  /**
   * @brief Claim memory for operand by WIC algorithm
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  void claim(const ir::OperandIndex &, size_t) override;
  /**
   * @brief Release memory for operand by WIC algorithm
   * @param[in] index The operand index
   */
  void release(const ir::OperandIndex &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override
  {
    if (!_initialized)
      buildMemoryPlans();
    return _capacity;
  }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  MemoryPlans &memory_plans() override;

private:
  void buildMemoryPlans();

  bool _initialized;
  uint32_t _capacity;
  MemoryPlans _mem_plans;
  std::unordered_set<ir::OperandIndex> _live_operands;
  ir::OperandIndexMap<std::unordered_set<ir::OperandIndex>> _interference_graph;
  // Sort operands by descending order of size
  std::multimap<uint32_t, ir::OperandIndex, std::greater<uint32_t>> _map_size_to_operands;
  std::multimap<uint32_t, ir::OperandIndex> _claim_table;
};

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_COMMON_MEMORY_PLANNER_H__
