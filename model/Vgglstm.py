import torch
from torch import nn
from torchvision import models

class VGGLSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_num_layers=2, num_channels=13, image_size=128):
        super(VGGLSTM, self).__init__()
        # Load a pre-trained VGG model
        vgg = models.vgg16(weights="IMAGENET1K_V1")
        
        # Modify the first convolutional layer to accept 13 channels
        self.features = list(vgg.features)
        self.features[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.features = nn.Sequential(*self.features)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.lstm = nn.LSTM(input_size=512 * 7 * 7, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        
        self.fc = nn.Linear(lstm_hidden_size, 1)  # Assuming predicting a single value

    def forward(self, x):
        batch_size, c, h, w = x.size()
        
        # Extract features with VGG
        x = self.features(x)
        x = self.global_avg_pool(x)

        x = x.view(batch_size, -1)
        x = x.unsqueeze(1) 
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out[:, -1, :]
        
        out = self.fc(lstm_out)
        
        return out


