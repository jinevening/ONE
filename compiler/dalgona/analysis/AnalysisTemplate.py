# Copyright 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A template for anlaysis code.
This template shows how to access information of each operataor inside the hooks.
Users can write their own analysis codes by modifying this file.
"""


class AnalysisTemplate(object):
    def StartAnalysis(self, args):
        """Called when the analysis starts"""
        print("Analysis started.")

    def EndAnalysis(self):
        """Called when the analysis ends"""
        print("Analysis ended.")

    def StartNetworkExecution(self, inputs):
        """Called when the execution of a network starts"""
        print("Network execution started.")

    def EndNetworkExecution(self, outputs):
        """Called when the execution of a network ends"""
        print("Network execution ended.")

    def DefaultOpPre(self, name, opcode, inputs):
        """Default hook called before an operator is executed"""
        print("name", name)
        print("opcode", opcode)
        print("inputs", inputs)

    def DefaultOpPost(self, name, opcode, inputs, output):
        """Default hook called after an operator is executed"""
        print("name", name)
        print("opcode", opcode)
        print("inputs", inputs)
        print("output", output)

    def Conv2DPre(self, name, input, filter, bias, padding, stride, dilation, fused_act):
        """Called before Conv2D layer execution"""
        print("name", name)
        print("input", input)
        print("filter", filter)
        print("bias", bias)
        print("padding", padding)
        print("stride", stride)
        print("dilation", dilation)
        print("fused activation", fused_act)

    def Conv2DPost(self, name, input, filter, bias, padding, stride, dilation, output,
                   fused_act):
        """Called after Conv2D layer execution"""
        print("name", name)
        print("input", input)
        print("filter", filter)
        print("bias", bias)
        print("padding", padding)
        print("stride", stride)
        print("dilation", dilation)
        print("output shape", output['data'].shape)
        print("output type", output['data'].dtype)
        print("fused activation", fused_act)
