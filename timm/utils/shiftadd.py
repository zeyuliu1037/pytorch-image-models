import torch
import numpy as np
import math
from torch.autograd import Function

import deepshift.kernels

def round_to_fixed(input, fraction=16, integer=16):
    assert integer >= 1, integer
    if integer == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -(fraction))
    bound = math.pow(2.0, integer-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta) * delta

    clipped_value = torch.clamp(rounded, min_val, max_val)
    return clipped_value

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)

    x_abs = torch.abs(x)
    shift = u_round(torch.log(x_abs) / np.log(2), rounding)

    return shift, sign

def round_power_of_2(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)
    # print(shift)
    x_rounded = (2.0 ** shift) * sign
    return x_rounded

def u_round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

class ConcWeight():
    def __init__(self, data=None, base=0, bits=8):
        self.data = data
        self.base = base
        self.bits = bits

##concatenate shift and sign together
def compress_bits(shift, sign):
    conc_weight = ConcWeight()

    if len(shift.shape) == 2:
        shift = shift.unsqueeze(-1).unsqueeze(-1)

    # if sign is ternary, then use a big shift value that is equivalent to multiplying by zero
    zero_sign_indices = (sign == 0).nonzero()
    shift[zero_sign_indices] = -32
    sign[zero_sign_indices] = +1

    conc_weight.bits = math.ceil(torch.log( - torch.min(shift) + 1)/ np.log(2))
    # treat shift to the right as the default
    shift = shift * -1
    minimum = int(torch.min(shift))
    if minimum < 0:
        conc_weight.base = minimum
        shift = shift - minimum
    else:
        conc_weight.base = 0

    num = int(32 / (conc_weight.bits + 1))
    row_length = int((shift.shape[1] * shift.shape[2] * shift.shape[3] + num -1) / num )
    size = row_length * shift.shape[0]

    conc_weight.data = deepshift.kernels.compress_sign_and_shift(shift.int().cuda(), sign.int().cuda(), size, conc_weight.base, conc_weight.bits, row_length, num)

    return conc_weight


def dynamic_range_for_sign(sign, threshold):
    # print(sign, threshold)
    with torch.no_grad():
        sign.data[sign.data < -threshold] = -1
        sign.data[sign.data > threshold] = 1
        sign.data[(-threshold <= sign.data) & (sign.data <= threshold)] = 0
    return sign

class myRoundFunction(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        return dynamic_range_for_sign(input, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def myround(input, threshold):
    return myRoundFunction.apply(input, threshold)


##############

class RoundPowerOf2(Function):
    @staticmethod
    def forward(ctx, input, stochastic=False):
        return round_power_of_2(input, stochastic)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def round_power_of_2(input, stochastic=False):
    return RoundPowerOf2.apply(input, stochastic)

class RoundFixedPoint(Function):
    @staticmethod
    def forward(ctx, input, quant_bits):
        return round_to_fixed(input, fraction=quant_bits)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def round_fixed_point(input, quant_bits):
    return RoundFixedPoint.apply(input, quant_bits)

class RoundFunction(Function):
    @staticmethod
    def forward(ctx, input, rounding='deterministic'):
        return u_round(input, rounding)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def round(input, rounding='deterministic'):
    return RoundFunction.apply(input, rounding)

class SignFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def sign(input):
    return SignFunction.apply(input)

class ClampFunction(Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def clamp(input, min, max):
    return ClampFunction.apply(input, min, max)

class ClampAbsFunction(Function):
    @staticmethod
    def forward(ctx, input, min, max):
        assert(min >= 0 and max >=0)

        input[input > max] = max
        input[input < -max] = -max

        input[(input > torch.zeros_like(input)) & (input < min)] = min
        input[(input < torch.zeros_like(input)) & (input > -min)] = -min
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def clampabs(input, min, max):
    return ClampAbsFunction.apply(input, min, max)

class LogFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.log(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def log(input):
    return LogFunction.apply(input)

class UnsymmetricGradMulFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        return torch.mul(input1, input2)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        return grad_output*input2, grad_output

def unsym_grad_mul(input1, input2):
    return UnsymmetricGradMulFunction.apply(input1, input2)


class AbsFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.abs(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def abs(input):
    return AbsFunction.apply(input)