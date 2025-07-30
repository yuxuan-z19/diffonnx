import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.onnx import ONNXProgram


# Reference: https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level2/86_Matmul_Divide_GELU.py
class Model(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.linear(x)
        x = x / self.divisor
        x = torch.nn.functional.gelu(x)
        return x


# From: https://github.com/deepreinforce-ai/CUDA-L1
class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.divisor = divisor

        scaled_weight = (self.weight / divisor).detach().clone()
        scaled_bias = (self.bias / divisor).detach().clone()

        self.register_buffer("scaled_weight_t", scaled_weight.t().contiguous())
        self.register_buffer("scaled_bias", scaled_bias.contiguous())

    def forward(self, x):
        out = torch.addmm(self.scaled_bias, x, self.scaled_weight_t)
        return torch.nn.functional.gelu(out)


# === Helper functions ===
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, output_size, divisor]


def get_programs() -> Tuple[ONNXProgram, ONNXProgram]:
    ref = Model(*get_init_inputs())
    usr = ModelNew(*get_init_inputs())
    inputs = get_inputs()

    ref_program = torch.onnx.export(ref, tuple(inputs), dynamo=True)
    usr_program = torch.onnx.export(usr, tuple(inputs), dynamo=True)

    ref_program.optimize()
    usr_program.optimize()

    return ref_program, usr_program
