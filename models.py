## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 224x224x1 to 220x220x32
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # 220x220x32 to 110x110x32
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 110x110x32 to 108x108x64
        self.conv2 = nn.Conv2d(32,64,3)
        
        # 108x108x64 to 54x54x64
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 54x54x64 to 52x52x128
        self.conv3 = nn.Conv2d(64,128,3)
        
        # 52x52x128 to 26x26x128
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 26x26x128 to 24x24x256 
        self.conv4 = nn.Conv2d(128,256,3)
        
        # 24x24x256 to 12x12x256
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # 12x12x256 to 12x12x512
        self.conv5 = nn.Conv2d(256,512,1)
        
        # 12x12x512 to 6x6x512
        self.pool5 = nn.MaxPool2d(2,2)
        
        # Dense
        self.fc1 = nn.Linear( 6 * 6 * 512 , 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)
        
        # Dropout for each layer
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.2)
        self.drop4 = nn.Dropout(p = 0.2)
        self.drop5 = nn.Dropout(p = 0.3)
        self.drop5_fc1 = nn.Dropout(p = 0.25)
        self.dropfc1_2 = nn.Dropout(p = 0.25)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.drop5_fc1(F.relu(self.fc1(x)))
        x = self.dropfc1_2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
