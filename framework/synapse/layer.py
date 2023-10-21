import torch
import torch.nn as nn

from typing import Union, Optional
from torch.nn.common_types import (_size_any_t, _size_1_t, _size_2_t)


def forward_with_time(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    ..code-block:: python
        B, T = 256, 100
        l1 = nn.Conv2d(1, 16, 3)
        l2 = nn.AvgPool2d(2, 2)
        out1 = forward_with_time(l1, torch.randn(B, T, 1, 28, 28))
        out2 = forward_with_time(l2, out1)

    """
    batch_size, steps = x.shape[:2]
    out = model(x.flatten(0, 1).contiguous())
    return out.view(batch_size, steps, *out.shape[1:])


class Layer(nn.Module):
    def __init__(self) -> None:
        super(Layer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input stimuli from pre-synapses in T time steps, shape=[N, T, D], while N is batch size,
        T is time step, D is feature dimension.

        :return: summation of pre-synapses stimuli to post-synapses neurons through synapse efficiency,
        each time step are integrated independently.
        """
        return forward_with_time(self.model, x)


class AdaptLayer(Layer):
    def __init__(self, model):
        """
        :param model: the specific computing model for a single step.

        .. code-block:: python
            N, T = 256, 100
            layer = GL(nn.Linear(784, 10))
            x = torch.randn(N, T, 784)
            out = layer(x)

        """
        super(AdaptLayer, self).__init__()
        self.model = model


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        """
        :param in_features: size of input stimuli each time step

        :param out_features: size of output spikes each time step

        :param bias: if set to ``False``, the layer will not learn an additive bias. Default: ``False``

        :param device: device for computation

        :param dtype: date type

        .. code-block:: python
            N, T = 256, 100
            linear = Linear(784, 10)
            x = torch.randn(N, T, 784)
            out = linear(x)

        """
        super(Linear, self).__init__()
        self.model = nn.Linear(in_features, out_features, bias, device, dtype)


class Conv1d(Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: Union[str, _size_1_t] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ) -> None:
        """
        :param in_channels(int): Number of channels in the input stimuli

        :param out_channels(int): Number of channels in the output stimuli

        :param kernel_size(int or tuple): Size of convolving kernel

        :param stride(int or tuple, optional): Stride of the convolution. Default: 1

        :param padding(int, tuple or str, optional): Padding added to both sides of the input. Default: 0:

        :param dilation: Spacing between kernel elements. Default: 1

        :param groups: Number of blocked connections from input channels to output channels. Default: 1

        :param bias: If ``True``, adds a learnable bias to the output. Default: ``False``

        :param padding_mode: `'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

        :param device:

        :param dtype:

        .. code-block:: python
            N, T = 256, 100
            conv1d = Conv1d(1, 4, 3)
            x = torch.randn(N, T, 1, 784)
            out = conv1d(x)
        """
        super(Conv1d, self).__init__()
        self.model = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)


class Conv2d(Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ) -> None:
        """
        :param in_channels(int): Number of channels in the input stimuli

        :param out_channels(int): Number of channels in the output stimuli

        :param kernel_size(int or tuple): Size of convolving kernel

        :param stride(int or tuple, optional): Stride of the convolution. Default: 1

        :param padding(int, tuple or str, optional): Padding added to both sides of the input. Default: 0:

        :param dilation: Spacing between kernel elements. Default: 1

        :param groups: Number of blocked connections from input channels to output channels. Default: 1

        :param bias: If ``True``, adds a learnable bias to the output. Default: ``False``

        :param padding_mode: `'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

        :param device:

        :param dtype:

        .. code-block:: python
            B, T = 256, 100
            conv2d = Conv2d(16, 32, 3)
            x = torch.randn(B, T, 16, 28, 28)
            out = conv2d(x)
        """
        super(Conv2d, self).__init__()
        self.model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)


class BatchNorm1d(Layer):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ) -> None:
        """
        :param num_features: number of simulate time steps (T)

        :param eps: a value added to the denominator for numerical stability. Default: 1e-5

        :param momentum: the value used for the running_mean and running_var computation. Can be set to ``None`` for
        cumulative moving average (i.e. simple average). Default: 0.1

        :param affine: a boolean value that when set to ``True``, this module ha learnable affine parameters. Default: ``True``

        :param track_running_stats: a boolean value that when set to ``True``, this module tracks the running mean and
        variance, and when set to ``False``, this module does not track such statistics, and initializes statistics buffers
        :attr:`running_mean` and :attr:`running_var` as ``None``. When these buffers are ``None``, this module always uses
        batch statistics. in both training and eval modes. Default: ``True``

        :param device:

        :param dtype:

        .. code-block:: python
            B, T = 256, 100
            bn1d = BatchNorm1d(1)
            x = torch.randn(B, T, 1, 784)
            out = bn1d(x)
        """
        super(BatchNorm1d, self).__init__()
        self.model = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class BatchNorm2d(Layer):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ) -> None:
        """
        :param num_features: number of simulate time steps (T)

        :param eps: a value added to the denominator for numerical stability. Default: 1e-5

        :param momentum: the value used for the running_mean and running_var computation. Can be set to ``None`` for
        cumulative moving average (i.e. simple average). Default: 0.1

        :param affine: a boolean value that when set to ``True``, this module ha learnable affine parameters. Default: ``True``

        :param track_running_stats: a boolean value that when set to ``True``, this module tracks the running mean and
        variance, and when set to ``False``, this module does not track such statistics, and initializes statistics buffers
        :attr:`running_mean` and :attr:`running_var` as ``None``. When these buffers are ``None``, this module always uses
        batch statistics. in both training and eval modes. Default: ``True``

        :param device:

        :param dtype:

        .. code-block:: python
            B, T = 256, 100
            bn1d = BatchNorm2d(1)
            x = torch.randn(B, T, 1, 28, 28)
            out = bn1d(x)
        """
        super(BatchNorm2d, self).__init__()
        self.model = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)


class MaxPool1d(Layer):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 padding: _size_any_t = 0, dilation: _size_any_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        """
        :param kernel_size: The size of the sliding window, must be > 0.

        :param stride: The stride of the sliding window, must be > 0. Default value is kernel_size.

        :param padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.

        :param dilation: The stride between elements within a sliding window, must be > 0.

        :param return_indices: Return the argmax along with the max values if True.

        :param ceil_mode: If True, will use ceil instead of floor to compute the output shape. This ensures that every
        element in the input tensor is covered by a sliding window.

        ..code-block:: python
            B, T = 256, 100
            maxpool1d = MaxPool1d(2)
            x = torch.randn(B, T, 784)
            out = maxpool1d(x)

        """
        super(MaxPool1d, self).__init__()
        self.model = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class MaxPool2d(Layer):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 padding: _size_any_t = 0, dilation: _size_any_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        """
        :param kernel_size: The size of the sliding window, must be > 0.

        :param stride: The stride of the sliding window, must be > 0. Default value is kernel_size.

        :param padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.

        :param dilation: The stride between elements within a sliding window, must be > 0.

        :param return_indices: Return the argmax along with the max values if True.

        :param ceil_mode: If True, will use ceil instead of floor to compute the output shape. This ensures that every
        element in the input tensor is covered by a sliding window.

        ..code-block:: python
            B, T = 256, 100
            maxpool2d = MaxPool2d(2)
            x = torch.randn(B, T, 28, 28)
            out = maxpool2d(x)

        """
        super(MaxPool2d, self).__init__()
        self.model = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class AvgPool1d(Layer):
    def __init__(self, kernel_size: _size_1_t, stride: _size_1_t = None, padding: _size_1_t = 0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True) -> None:
        """
        :param kernel_size:  the size of the window.

        :param stride: the stride of the window. Default value is kernel_size.

        :param padding: implicit zero padding to be added on both sides.

        :param ceil_mode: when True, will use ceil instead of floor to compute the output shape.

        :param count_include_pad: when True, will include the zero-padding in the averaging calculation.

        ..code-block:: python
            out = AvgPool1d(2, 2)(torch.randn(B, T, 1, 784))

        """
        super(AvgPool1d, self).__init__()
        self.model = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)


class AvgPool2d(Layer):
    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True,
                 divisor_override: Optional[int] = None) -> None:
        """
        :param kernel_size:  the size of the window.

        :param stride: the stride of the window. Default value is kernel_size.

        :param padding: implicit zero padding to be added on both sides.

        :param ceil_mode: when True, will use ceil instead of floor to compute the output shape.

        :param count_include_pad: when True, will include the zero-padding in the averaging calculation.

        :param divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.

        ..code-block:: python
            out = AvgPool2d(2, 2)(torch.randn(B, T, 1, 28, 28))

        """
        super(AvgPool2d, self).__init__()
        self.model = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
