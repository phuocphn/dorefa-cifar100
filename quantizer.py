import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
      weight_q = 2 * self.uniform_q(weight) - 1
    return weight_q

  def extra_repr(self):
    return "w_bit=%s"%(self.w_bit)


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q
  def extra_repr(self):
    return "a_bit=%s"%(self.a_bit)


class DirectQuant(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, qmax, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output = ((output * qmax).round()) / qmax
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None

class PACTClippingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clipping_value, bit):
        ctx.save_for_backward(input, clipping_value)
        x_tilder = torch.clamp(input, 0, clipping_value.data)
        param_in = x_tilder / clipping_value
        ctx.param_in = param_in

        output = DirectQuant.apply(param_in, 2 ** bit -1)
        ctx.output = output
        return output * clipping_value

    @staticmethod
    def backward(ctx, grad_output):
        input, clipping_value, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_clipping_value = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[input>=clipping_value] = 0

        grad_clipping_value[input<clipping_value] = 0
        return grad_input, torch.sum(grad_clipping_value), None # correct version


class pact_quantize_fn(nn.Module):
    def __init__(self, init_alpha=8.0, a_bit=4):
        super(pact_quantize_fn, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.bit = a_bit

    def forward(self, input):
        return PACTClippingFunc.apply(input, self.alpha, self.bit)

    def extra_repr(self):
        return "bits=%s" % (self.bit)


class DRF_QConv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1, bias=True, bit=4):
    super(DRF_QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups, bias)
    self.bit = bit
    self.weight_quantize_fn = weight_quantize_fn(w_bit=bit)
    self.activation_quantize_fn = pact_quantize_fn(a_bit = bit)

  def forward(self, input, order=None):
    weight_q = self.weight_quantize_fn(self.weight)
    input_q = self.activation_quantize_fn(input)
    return F.conv2d(input_q, weight_q, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)



class DRF_QLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    
    super(DRF_QLinear, self).__init__(in_features, out_features, bias)
    self.bit = bit
    self.weight_quantize_fn = weight_quantize_fn(w_bit=bit)
    self.activation_quantize_fn = pact_quantize_fn(a_bit = bit)

  def forward(self, input):
    weight_q = self.weight_quantize_fn(self.weight)
    input_q = self.activation_quantize_fn(input)
    return F.linear(input_q, weight_q, self.bias)



# if __name__ == '__main__':
#   import numpy as np
#   import matplotlib.pyplot as plt

#   a = torch.rand(1, 3, 32, 32)

#   Conv2d = conv2d_Q_fn(w_bit=2)
#   conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
#   act = activation_quantize_fn(a_bit=3)

#   b = conv(a)
#   b.retain_grad()
#   c = act(b)
#   d = torch.mean(c)
#   d.retain_grad()

#   d.backward()
#   pass