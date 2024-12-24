from torch import Tensor, nn

class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 activation: str = "ReLU"):
        super().__init__()
        # kernel_size = 3: 3x3 convolution
        # For odd kernel_size, edges will be removed without padding, causing some of the weights to not show in the output
        # padding <= kernel_size // 2, any more than that will cause the edges to show "-inf" values
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2, bias = False)
        # Set track_running_stats = False for all batch normalizations so that it will always use batch statistics to normalize the activations
        self.normalization = nn.BatchNorm2d(out_channels, track_running_stats = False)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        y = self.convolution(x)
        y = self.normalization(y)
        y = self.activation(y)
        return y

class ProjectionShortcut(nn.Module):
    # Identity Shortcut vs Projection Shortcut: https://www.researchgate.net/figure/Resnet-structure-consisting-of-a-identity-shortcut-and-b-projection-shortcut-Identity_fig5_364437920
    # Attempts to project the residual features to the correct size, as x and F(x) are of different dimensions (see Equation 2 of ResNet paper)
    # Can also be used in bottleneck to downsample input via stride = 2
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 2):
        super().__init__()
        # kernel_size = 1: 1x1 convolution
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
        self.normalization = nn.BatchNorm2d(out_channels, track_running_stats = False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.convolution(x)
        y = self.normalization(y)
        return y
