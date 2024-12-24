from .blocks import BasicBlock, BottleNeck
from .resnet import ResNet

ResNet18 = lambda: ResNet(blocks = [BasicBlock for _ in range(4)], depths = [2, 2, 2, 2])
ResNet34 = lambda: ResNet(blocks = [BasicBlock for _ in range(4)], depths = [3, 4, 6, 3])
ResNet50 = lambda: ResNet(blocks = [BottleNeck for _ in range(4)], depths = [3, 4, 6, 3])
ResNet101 = lambda: ResNet(blocks = [BottleNeck for _ in range(4)], depths = [3, 4, 23, 3])
ResNet152 = lambda: ResNet(blocks = [BottleNeck for _ in range(4)], depths = [3, 4, 36, 3])
