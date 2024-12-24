import os
import torch
from .layers import ConvLayer, ProjectionShortcut
from .blocks import BasicBlock, BottleNeck
from typing import List, Union
from torch import Tensor, nn
from safetensors.torch import load_model, save_model

class ResNet(nn.Module):
    # reduction_scale and downsample_3x3 will only be applied for BottleNeck
    reduction_scale = 4
    def __init__(self,
                 num_channels: int = 3,
                 in_channels: int = 64,
                 blocks: List[Union[BasicBlock, BottleNeck]] = [BottleNeck for _ in range(4)],
                 depths: List[int] = [3, 4, 6, 3],
                 downsample_first_stage: bool = True,
                 downsample_3x3: bool = True):
        assert len(blocks) == len(depths) > 0 # number of stages; depth refers to number of layers at each stage
        super().__init__()
        self.embedder = ConvLayer(num_channels, in_channels, kernel_size = 7, stride = 2, activation = "ReLU")
        self.max_pooler = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        # In the first stage of ResNet, downsampling via stride = 2 is better than pooling because it compresses three 3x3 convolutions into one 7x7 convolution
        # This significantly reduces the amount of computation needed to be done in the subsequent stages
        out_channels = in_channels * self.reduction_scale
        self.stages = nn.ModuleList([
            nn.Sequential(
                blocks[0](in_channels, out_channels, stride = 2 if downsample_first_stage is True else 1, activation = "ReLU", reduction_scale = self.reduction_scale, downsample_3x3 = downsample_3x3),
                *[blocks[0](out_channels, out_channels, stride = 1, activation = "ReLU", reduction_scale = self.reduction_scale, downsample_3x3 = downsample_3x3) for _ in range(depths[0])]
            )
        ])
        for i in range(1, len(depths)):
            in_channels = out_channels
            out_channels *= 2 # double the number of channels that are being output
            stage = nn.Sequential(
                blocks[i](in_channels, out_channels, stride = 2, activation = "ReLU", reduction_scale = self.reduction_scale, downsample_3x3 = downsample_3x3),
                *[blocks[i](out_channels, out_channels, stride = 1, activation = "ReLU", reduction_scale = self.reduction_scale, downsample_3x3 = downsample_3x3) for _ in range(depths[i] - 1)]
            )
            self.stages.append(stage)
        self.out_channels = out_channels
        self.avg_pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.apply(self._init_weights)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
        elif isinstance(module, nn.BatchNorm2d): # constant weight for batch normalization layer: https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def save(self, file_path: Union[str, os.PathLike]) -> None:
        save_model(self, file_path)

    def load(self, file_path: Union[str, os.PathLike]) -> None:
        load_model(self, file_path)

    def forward(self, x: Tensor) -> Tensor:
        y = self.embedder(x)
        y = self.max_pooler(y)
        for stage in self.stages:
            y = stage(y)
        y = self.avg_pooler(y)
        return y
