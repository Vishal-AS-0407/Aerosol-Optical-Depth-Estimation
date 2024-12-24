import torch
import torch.nn as nn
from timm.models import create_model
from model import vanillanet

class VanillaPred(nn.Module):
    def __init__(self):
        super(VanillaPred, self).__init__()
        # Load a pre-trained VanillaNet model
        vanillanet_model = create_model('vanillanet_10')

        # Modify the first convolutional layer to accept 13 channels
        self.vanillanet = torch.nn.Sequential(*list(vanillanet_model.children()))
        self.vanillanet[0] = nn.Sequential(
            nn.Conv2d(13, 512, kernel_size=(4, 4), stride=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-06, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.vanillanet[-1] = nn.Sequential(nn.Conv2d(1000,100,kernel_size=(1,1),stride=(1,1)),
                                            nn.BatchNorm2d(100, eps=1e-06, momentum=0.1, affine=True, track_running_stats=True))
        self.conv1 = nn.Conv2d(100,1,kernel_size=(1,1),stride=(1,1))
        # self.fc = nn.Linear(1000,1000)# Adjust the output features according to your needs

    def forward(self, x):
        for i in self.vanillanet:
            #print(f'{i}\n.................................\n')
            if isinstance(i,nn.ModuleList):
                for j in i:
                    x = j(x)
                continue
            x = i(x)
        x = self.conv1(x)
        return x[:,0,0,0]