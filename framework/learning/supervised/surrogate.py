import torch


lens = 1.0

class Rect_Actfun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # V_th = torch.tensor(V_th)
        ctx.save_for_backward(input)
        return input.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()
        # temp = 1 * (lens - input.abs()).clamp(min=0.0)
        # return grad_input * temp

rect_fn = Rect_Actfun.apply

class triangle_Actfun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # V_th = torch.tensor(V_th)
        ctx.save_for_backward(input)
        return input.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()
        # temp = 1 * (lens - (input - V_th).abs()).clamp(min=0.0)
        return grad_input * temp

tri_fn = triangle_Actfun.apply


class SLayer_Actfun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_out):
        # f = alpha * exp(-beta * |u - v_th|)
        alpha = 1
        beta = 1
        input, = ctx.saved_tensors
        return grad_out * alpha * torch.exp(-beta * torch.abs(input))

class Sigmoid_Actfun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_out):
        input, = ctx.saved_tensors
        return grad_out * torch.sigmoid(input) * (1 - torch.sigmoid(input))

sigmoid_fn = Sigmoid_Actfun.apply

slayer_fn = SLayer_Actfun.apply

