from .layers import ConvLayer, ProjectionShortcut
from torch import Tensor, nn

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 activation: str = "ReLU",
                 **kwargs):
        super().__init__()
        if in_channels != out_channels or stride != 1: # projection shortcut
            self.shortcut = ProjectionShortcut(in_channels, out_channels, stride = stride)
        else: # identity shortcut
            self.shortcut = nn.Identity()
        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size = 3, stride = stride, activation = "ReLU"),
            ConvLayer(out_channels, out_channels, kernel_size = 3, stride = 1, activation = "Identity") # when activation = "Identity", it can be said that there is no activation function
        )
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        y, residual = self.layers(x), self.shortcut(x)
        y += residual
        y = self.activation(y)
        return y

class BottleNeck(nn.Module):
    # Bottleneck Residual Block is a variant of the Residual Block
    # Instead of using two 3x3 convolutions, Bottleneck uses two 1x1 convolutions before and after a 3x3 convolution to create a bottleneck
    # This reduces the number of matrix multiplication operations needed, allowing ResNets to become deeper while having less parameters
    # Generally there are two methods for downsampling: striding in convolution and pooling after convolution
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 activation: str = "ReLU",
                 reduction_scale: int = 4,
                 downsample_3x3: bool = True,
                 **kwargs):
        super().__init__()
        if in_channels != out_channels or stride != 1: # projection shortcut
            self.shortcut = ProjectionShortcut(in_channels, out_channels, stride = stride)
        else: # identity shortcut
            self.shortcut = nn.Identity()
        reduced_channels = out_channels // reduction_scale
        # Downsampling can either be done in the first 1x1 convolution or the 3x3 convolution
        # Note that if your stride = 1, downsample_3x3 is ignored
        self.layers = nn.Sequential(
            ConvLayer(in_channels, reduced_channels, kernel_size = 1, stride = stride if downsample_3x3 is False else 1, activation = "ReLU"),
            ConvLayer(reduced_channels, reduced_channels, kernel_size = 3, stride = stride if downsample_3x3 is True else 1, activation = "ReLU"),
            ConvLayer(reduced_channels, out_channels, kernel_size = 1, stride = 1, activation = "Identity")
        )
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        y, residual = self.layers(x), self.shortcut(x)
        y += residual
        y = self.activation(y)
        return y
