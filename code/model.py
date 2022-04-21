import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datamodel import CircleDataset
import os


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=0),  
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class autoencoder_2(nn.Module):

    def __init__(self):
        super(autoencoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=3, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(4, 2, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 4, 3, stride=2),  
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 2, 5, stride=3, padding=0),  
            nn.ReLU(True),
            nn.ConvTranspose2d(2, 1, 2, stride=2, padding=1),  
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x