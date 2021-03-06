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

#ifndef __LUCI_IMPORT_NODES_H__
#define __LUCI_IMPORT_NODES_H__

#include "Nodes/CircleAbs.h"
#include "Nodes/CircleAdd.h"
#include "Nodes/CircleArgMax.h"
#include "Nodes/CircleAveragePool2D.h"
#include "Nodes/CircleBatchToSpaceND.h"
#include "Nodes/CircleCast.h"
#include "Nodes/CircleCustom.h"
#include "Nodes/CircleConcatenation.h"
#include "Nodes/CircleConst.h"
#include "Nodes/CircleConv2D.h"
#include "Nodes/CircleCos.h"
#include "Nodes/CircleDepthwiseConv2D.h"
#include "Nodes/CircleDiv.h"
#include "Nodes/CircleElu.h"
#include "Nodes/CircleEqual.h"
#include "Nodes/CircleExp.h"
#include "Nodes/CircleExpandDims.h"
#include "Nodes/CircleFill.h"
#include "Nodes/CircleFloorDiv.h"
#include "Nodes/CircleFloorMod.h"
#include "Nodes/CircleFullyConnected.h"
#include "Nodes/CircleGather.h"
#include "Nodes/CircleGatherNd.h"
#include "Nodes/CircleGreater.h"
#include "Nodes/CircleGreaterEqual.h"
#include "Nodes/CircleIf.h"
#include "Nodes/CircleLeakyRelu.h"
#include "Nodes/CircleLess.h"
#include "Nodes/CircleLogicalAnd.h"
#include "Nodes/CircleLogicalNot.h"
#include "Nodes/CircleLogicalOr.h"
#include "Nodes/CircleLogistic.h"
#include "Nodes/CircleMaximum.h"
#include "Nodes/CircleMaxPool2D.h"
#include "Nodes/CircleMean.h"
#include "Nodes/CircleMinimum.h"
#include "Nodes/CircleMirrorPad.h"
#include "Nodes/CircleMul.h"
#include "Nodes/CircleNotEqual.h"
#include "Nodes/CircleOneHot.h"
#include "Nodes/CirclePack.h"
#include "Nodes/CirclePad.h"
#include "Nodes/CirclePow.h"
#include "Nodes/CircleRange.h"
#include "Nodes/CircleReduceAny.h"
#include "Nodes/CircleReduceMax.h"
#include "Nodes/CircleReduceProd.h"
#include "Nodes/CircleRelu.h"
#include "Nodes/CircleReluN1To1.h"
#include "Nodes/CircleReshape.h"
#include "Nodes/CircleRsqrt.h"
#include "Nodes/CircleSelect.h"
#include "Nodes/CircleShape.h"
#include "Nodes/CircleSin.h"
#include "Nodes/CircleSlice.h"
#include "Nodes/CircleSoftmax.h"
#include "Nodes/CircleSpaceToBatchND.h"
#include "Nodes/CircleSpaceToDepth.h"
#include "Nodes/CircleSplit.h"
#include "Nodes/CircleSplitV.h"
#include "Nodes/CircleSquare.h"
#include "Nodes/CircleSquaredDifference.h"
#include "Nodes/CircleSqueeze.h"
#include "Nodes/CircleStridedSlice.h"
#include "Nodes/CircleSub.h"
#include "Nodes/CircleSum.h"
#include "Nodes/CircleTanh.h"
#include "Nodes/CircleTile.h"
#include "Nodes/CircleTopKV2.h"
#include "Nodes/CircleTranspose.h"
#include "Nodes/CircleUnpack.h"
#include "Nodes/CircleWhile.h"
#include "Nodes/CircleZerosLike.h"

#endif // __LUCI_IMPORT_NODES_H__
