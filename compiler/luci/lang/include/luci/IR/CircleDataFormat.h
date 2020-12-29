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

#ifndef __LUCI_IR_CIRCLEDATAFORMAT_H__
#define __LUCI_IR_CIRCLEDATAFORMAT_H__

namespace luci
{

enum CircleDataFormat
{
  UNDEFINED, // This is not defined by Circle schema. This was added to prevent programming error.

  // For 2D data, NHWC(batch, height, width, channels)
  // For 3D data, NDHWC(batch, depth, height, width, channels)
  CHANNELS_LAST,
  // For 2D data, NCHW(batch, channels, height, width)
  // For 3D data, NCDHW(batch, channels, depth, height, width)
  CHANNELS_FIRST,
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEDATAFORMAT_H__
