import torch
from torch import nn
from torchvision import models

class ResNetLSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_num_layers=2, num_channels=13, image_size=128):
        super(ResNetLSTM, self).__init__()
        # Load a pre-trained ResNet model
        resnet = models.resnet18(weights="IMAGENET1K_V1")

        # Modify the first convolutional layer to accept 13 channels
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet[0] = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)

        # Fully connected layer to predict the final output
        self.fc = nn.Linear(lstm_hidden_size, 1)  # Assuming predicting a single value

    def forward(self, x, for_summary=False):
        batch_size = x.size()[0]
        
        # Extract features with ResNet
        x = self.resnet(x)
        x = self.global_avg_pool(x)
        
        # Reshape for LSTM
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output of the LSTM (many-to-one)
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layer to predict the value
        out = self.fc(lstm_out)
        
        return out