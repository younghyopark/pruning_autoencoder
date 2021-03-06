import torch
import torch.nn as nn
import torch.nn.functional as F

class Fully_Connected_AE(nn.Module):
    def __init__(self, x_dim, dimensions, sigmoid, bias=True, alternative = False):
        super(Fully_Connected_AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.encoder = Encoder(x_dim, dimensions, bias)
        # decoder part
        self.alternative = alternative
        self.decoder = Generator(x_dim, dimensions, sigmoid, bias)
        if self.alternative:
            self.bottleneck = Bottleneck(x_dim, dimensions, bias)

    def recon_error(self, x):
        z = self.encoder(x)
        if self.alternative:
            z = self.bottleneck(z)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        if self.alternative:
            z = self.bottleneck(z)
        return self.decoder(z)


class Convolutional_AE(nn.Module):
    def __init__(self, x_dim, dimensions, sigmoid, bias=True, alternative = False):
        super(Fully_Connected_AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.encoder = ConvEncoder(x_dim, dimensions, bias)
        # decoder part
        self.alternative = alternative
        self.decoder = ConvGenerator(x_dim, dimensions, sigmoid, bias)
        if self.alternative:
            self.bottleneck = Bottleneck(x_dim, dimensions, bias)

    def recon_error(self, x):
        z = self.encoder(x)
        if self.alternative:
            z = self.bottleneck(z)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        if self.alternative:
            z = self.bottleneck(z)
        return self.decoder(z)

    
class Encoder(nn.Module):
    def __init__(self, x_dim, dimensions, bias):
        super(Encoder, self).__init__()
        self.dimensions = dimensions
        self.fc1 = nn.Linear(x_dim, dimensions[0],bias)
        self.fc2 = nn.Linear(dimensions[0], dimensions[1],bias)
        if dimensions[2]>0:
            self.fc3 = nn.Linear(dimensions[1], dimensions[2],bias)
        if dimensions[3]>0:
            self.fc4 = nn.Linear(dimensions[2],dimensions[3],bias)
        if dimensions[4] >0:
            self.fc5 = nn.Linear(dimensions[3],dimensions[4],bias)
        if dimensions[5] >0:
            self.fc6 = nn.Linear(dimensions[4],dimensions[5],bias)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        if self.dimensions[5] >0:
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc5(h))
            h = self.fc6(h)
        elif self.dimensions[4] >0:
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = self.fc5(h)
        elif self.dimensions[3] >0:
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
        elif self.dimensions[2] >0:
            h = F.relu(self.fc2(h))
            h = self.fc3(h)
        else:
            h = self.fc2(h)
        return h
    
    
class Generator(nn.Module):
    def __init__(self, x_dim, dimensions, sigmoid,bias):
        super(Generator, self).__init__()
        self.dimensions = dimensions
        self.sigmoid = sigmoid
        if dimensions[5] >0:
            self.fc6 = nn.Linear(dimensions[5],dimensions[4],bias)
        if dimensions[4] >0:
            self.fc5 = nn.Linear(dimensions[4],dimensions[3],bias)
        if dimensions[3] >0:
            self.fc4 = nn.Linear(dimensions[3],dimensions[2],bias)
        if dimensions[2] >0:
            self.fc3 = nn.Linear(dimensions[2], dimensions[1],bias)
        self.fc2 = nn.Linear(dimensions[1], dimensions[0],bias)
        self.fc1 = nn.Linear(dimensions[0], x_dim,bias)
    
    def forward(self, z):
        if self.dimensions[5] >0:
            h = F.relu(self.fc6(z))
            h = F.relu(self.fc5(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif self.dimensions[4] >0:
            h = F.relu(self.fc5(z))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif self.dimensions[3]>0:
            h = F.relu(self.fc4(z))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif self.dimensions[2]>0:
            h = F.relu(self.fc3(z))
            h = F.relu(self.fc2(h))
        else:
            h = F.relu(self.fc2(z))
        h = self.fc1(h)
        if self.sigmoid:
            h = torch.sigmoid(h)
        return h


class ConvEncoder(nn.Module):
    def __init__(self, x_dim, dimensions, bias):
        super(Encoder, self).__init__()
        self.dimensions = dimensions
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # 16 * 16 * 16
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # 16* 8 * 8
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # 16* 4* 4
        self.fc3 = nn.Linear(256,64)
        self.fc4 = nn.Linear(64,32)
    
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    
class ConvGenerator(nn.Module):
    def __init__(self, x_dim, dimensions, sigmoid,bias):
        super(Generator, self).__init__()
        self.dimensions = dimensions
        self.sigmoid = sigmoid
        if dimensions[5] >0:
            self.fc6 = nn.Linear(dimensions[5],dimensions[4],bias)
        if dimensions[4] >0:
            self.fc5 = nn.Linear(dimensions[4],dimensions[3],bias)
        if dimensions[3] >0:
            self.fc4 = nn.Linear(dimensions[3],dimensions[2],bias)
        if dimensions[2] >0:
            self.fc3 = nn.Linear(dimensions[2], dimensions[1],bias)
        self.fc2 = nn.Linear(dimensions[1], dimensions[0],bias)
        self.fc1 = nn.Linear(dimensions[0], x_dim,bias)
    
    def forward(self, z):
        if self.dimensions[5] >0:
            h = F.relu(self.fc6(z))
            h = F.relu(self.fc5(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif self.dimensions[4] >0:
            h = F.relu(self.fc5(z))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif self.dimensions[3]>0:
            h = F.relu(self.fc4(z))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif self.dimensions[2]>0:
            h = F.relu(self.fc3(z))
            h = F.relu(self.fc2(h))
        else:
            h = F.relu(self.fc2(z))
        h = self.fc1(h)
        if self.sigmoid:
            h = torch.sigmoid(h)
        return h



class Bottleneck(nn.Module):
    def __init__(self, x_dim, dimensions,bias):
        super(Bottleneck, self).__init__()
        self.dimensions = dimensions
        # if dimensions[5] >0:
        #     self.fc6 = nn.Linear(dimensions[5],dimensions[4],bias)
        # if dimensions[4] >0:
        #     self.fc5 = nn.Linear(dimensions[4],dimensions[3],bias)
        # if dimensions[3] >0:
        #     self.fc4 = nn.Linear(dimensions[3],dimensions[2],bias)
        # if dimensions[2] >0:
        #     self.fc3 = nn.Linear(dimensions[2], dimensions[1],bias)
        self.bottleneck = nn.Linear(dimensions[3], dimensions[3],False)
        # self.fc1 = nn.Linear(dimensions[0], x_dim,bias)
    
    def forward(self, z):
        h = F.relu(self.bottleneck(z))
        # if self.sigmoid:
        #     h = torch.sigmoid(h)
        return h