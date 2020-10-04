import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd.function import InplaceFunction, Function
from functools import partial

# def uniform_quantize(k):
#   class qfn(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input):
#       if k == 32:
#         out = input
#       elif k == 1:
#         out = torch.sign(input)
#       else:
#         n = float(2 ** k - 1)
#         out = torch.round(input * n) / n
#       return out

#     @staticmethod
#     def backward(ctx, grad_output):
#       grad_input = grad_output.clone()
#       return grad_input

#   return qfn().apply


# class weight_quantize_fn(nn.Module):
#   def __init__(self, w_bit):
#     super(weight_quantize_fn, self).__init__()
#     assert w_bit <= 8 or w_bit == 32
#     self.w_bit = w_bit
#     self.uniform_q = uniform_quantize(k=w_bit)

#   def forward(self, x):
#     if self.w_bit == 32:
#       weight_q = x
#     elif self.w_bit == 1:
#       E = torch.mean(torch.abs(x)).detach()
#       weight_q = self.uniform_q(x / E) * E
#     else:
#       weight = torch.tanh(x)
#       weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
#       weight_q = 2 * self.uniform_q(weight) - 1
#     return weight_q

#   def extra_repr(self):
#     return "w_bit=%s"%(self.w_bit)


# class activation_quantize_fn(nn.Module):
#   def __init__(self, a_bit):
#     super(activation_quantize_fn, self).__init__()
#     assert a_bit <= 8 or a_bit == 32
#     self.a_bit = a_bit
#     self.uniform_q = uniform_quantize(k=a_bit)

#   def forward(self, x):
#     if self.a_bit == 32:
#       activation_q = x
#     else:
#       activation_q = self.uniform_q(torch.clamp(x, 0, 1))
#       # print(np.unique(activation_q.detach().numpy()))
#     return activation_q
#   def extra_repr(self):
#     return "a_bit=%s"%(self.a_bit)


# class DRF_QConv2d(nn.Conv2d):
#   def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                padding=0, dilation=1, groups=1, bias=True, bit=4):
#     super(DRF_QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                    padding, dilation, groups, bias)
#     self.bit = bit
#     self.weight_quantize_fn = weight_quantize_fn(w_bit=bit)
#     self.activation_quantize_fn = activation_quantize_fn(a_bit = bit)

#   def forward(self, input, order=None):
#     weight_q = self.weight_quantize_fn(self.weight)
#     input_q = self.activation_quantize_fn(input)
#     return F.conv2d(input_q, weight_q, self.bias, self.stride,
#                     self.padding, self.dilation, self.groups)



# class DRF_QLinear(nn.Linear):
#   def __init__(self, in_features, out_features, bias=True):
    
#     super(DRF_QLinear, self).__init__(in_features, out_features, bias)
#     self.bit = bit
#     self.weight_quantize_fn = weight_quantize_fn(w_bit=bit)
#     self.activation_quantize_fn = activation_quantize_fn(a_bit = bit)

#   def forward(self, input):
#     weight_q = self.weight_quantize_fn(self.weight)
#     input_q = self.activation_quantize_fn(input)
#     return F.linear(input_q, weight_q, self.bias)

class StandardQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bit):
        input = input.clone()
        return  ((input * qmax).round()) /  2**bit - 1
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None


class ThresholdAlignmentQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bit):
        input = input.clone()
        output = ((2**bit) * input - 0.5).round().clamp(0, 2**bit - 1)
        output = output / (2**bit -1)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None

class QuantizableDNNS_QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, quant=False, quant_scale=False, bit=4):
        super(QuantizableDNNS_QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.bit = bit
        self.quantize_fn = ThresholdAlignmentQuantizer.apply

    def forward(self, input):
        if self.bit == 32:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)

        quantized_input = self.quantize_fn(input.clamp(0, 1), self.bit)
        w = torch.tanh(self.weight)
        max_w = torch.max(torch.abs(w)).detach()
        w = w / (2*max_w) + 0.5

        quantized_weight = 2 * self.quantize_fn(w, self.bit) -1
        return F.conv2d(quantized_input, quantized_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return super(QuantizableDNNS_QConv2d, self).extra_repr() + ", bit=%s" % (self.bit)

class QuantizableDNNS_QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, num_bits=8, quant=False, quant_scale=False, bit=4):
        super(QuantizableDNNS_QLinear, self).__init__(in_features, out_features, bias)
        self.bit = bit
        self.quantize_fn = ThresholdAlignmentQuantizer.apply


    def forward(self, input):
        if self.bit == 32:
            return F.linear(input, self.weight, self.bias)

        assert (self.bit != -1), "(linear) invalid current bit-width."
        quantized_input = self.quantize_fn(input.clamp(0, 1), self.bit)

        w = torch.tanh(self.weight)
        max_w = torch.max(torch.abs(w)).detach()
        w = w / (2*max_w) + 0.5

        quantized_weight = 2 * self.quantize_fn(w, self.bit) -1
        return F.linear(quantized_input, quantized_weight, self.bias)

    def extra_repr(self):
        return super(QuantizableDNNS_QLinear, self).extra_repr() + ", bit=%s" % (self.bit)
    # def extra_repr(self):
    #     return "bit=%s" % (self.bit)