"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, labels)
    return loss

class ResNet_block(nn.Module):
    """
        The class defines the basic block that needs to be used in ResNet architecture.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample_layer = nn.Sequential()
        if (stride != 1) or (out_channels != in_channels):
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()
        

    def forward(self, xb):
        identity = xb

        xb = self.conv1(xb)
        xb = self.conv2(xb)

        identity = self.downsample_layer(identity)
        
        xb += identity
        xb = self.relu(xb)
        return xb


class ResNeXt_block(nn.Module):
    """
        The class defines the basic block that needs to be used in ResNeXt architecture.
    """
    def __init__(self, in_channels, out_channels, bottleneck_width=4, stride=1, cardinality=32):
        super().__init__()

        intermediate_channels = bottleneck_width * cardinality
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample_layer = nn.Sequential()
        if (stride != 1) or (out_channels != in_channels):
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, xb):
        identity = xb

        xb = self.conv1(xb)
        xb = self.conv2(xb)
        xb = self.conv3(xb)

        identity = self.downsample_layer(identity)

        xb += identity
        return xb


class DenseNet_block(nn.Module):
    """
        The class defines the basic block that needs to be used in DenseNet architecture.
    """
    def __init__(self, in_channels, growth_rate, bottleneck_width=4):
        super().__init__()

        inner_channel = bottleneck_width * growth_rate

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], 1)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))



class CIFAR10Model(ImageClassificationBase):
    def __init__(self, model_type):
        #############################
        # Fill your network initialization of choice here!
        #############################
        super().__init__()
        self.model_type = model_type

        if model_type == 1:
            self.basicnet = self.init_BasicNet()
        elif model_type == 2:
            self.batchnormnet = self.init_BatchNormNet()
        elif model_type == 3:
            self.init_ResNet(ResNet_block)
        elif model_type == 4:
            self.init_ResNeXt(ResNeXt_block)
        elif model_type == 5: 
            self.init_DenseNet(DenseNet_block)


    ##############################################################  Basic Network  #################################################################
    def init_BasicNet(self):
        """
            The initialization of my implementation of a basic neural network for CIFAR-10 classification.
        """

        # input size: 32 x 32 x 3
        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),    # output size: 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # output size: 16 x 16 x 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),    #output size: 16 x 16 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # output size: 8 x 8 x 64
            nn.Flatten(),
            nn.Linear(in_features=8*8*64, out_features=100),
            nn.Linear(in_features=100, out_features=10)
        )
    

    ############################################################  Improved Network  ###############################################################
    def init_BatchNormNet(self):
        """
            The initialization of an improved neural network compared to the basic network to get an 
            higher accuracy for CIFAR-10 classification. This improvement is achieved by adding more
            batch normalization layers throughout the network.
        """

        # input size: 32 x 32 x 3
        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),    
            nn.BatchNorm2d(num_features=32),    # output size: 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # output size: 16 x 16 x 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),    
            nn.BatchNorm2d(num_features=64),    #output size: 16 x 16 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # output_size = 8 x 8 x 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),    
            nn.BatchNorm2d(num_features=128),    # output size: 8 x 8 x 128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # output size: 4 x 4 x 128
            nn.Flatten(),
            nn.Linear(in_features=4*4*128, out_features=150),
            nn.Linear(in_features=150, out_features=10)
        )


    ###############################################################  ResNet  ##########################################################################
    def init_ResNet(self, block, num_classes=10):
        """
            This is the initialization of a Residual Network (ResNet). It consists of a convoloutional layer that
            converts the 3-channel image data to a 16-channel data. Then ResNet layers containing ResNet 
            blocks are added to the network according to the "n_blocks" variable.The output of these layers are 
            passed through an AdaptiveAvgPool layer and then flattened to be passed through two fully 
            connected layers. 
        """
        
        n_blocks = [3, 4, 6, 3]
        in_channels = 16
        out_channels = 2 * in_channels
        stride = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # network is created in this format for better visualization in tensorboard
        self.ResNet_layers = nn.Sequential()
        self.ResNet_layers.add_module('ResNet_layer_1', self.make_ResNet_layer(block, in_channels, out_channels, n_blocks[0], stride=1))
        in_channels *= 2

        for i in range(1, len(n_blocks)):
            out_channels = 2 * in_channels
            self.ResNet_layers.add_module(f'ResNet_layer_{i+1}', self.make_ResNet_layer(block, in_channels, out_channels, n_blocks[i], stride))
            in_channels *= 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(out_channels, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
    def make_ResNet_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        ResNet_layer = nn.Sequential()
        ResNet_layer.add_module('ResNet_block_1', block(in_channels, out_channels, stride))
        self.inner_channels = out_channels
        for i in range(num_blocks - 1):
            ResNet_layer.add_module(f'ResNet_block_{i+2}', block(out_channels, out_channels))
            
        return ResNet_layer

    def ResNet_forward(self, xb):
        xb = self.conv1(xb)
        xb = self.ResNet_layers(xb)
        xb = self.avgpool(xb)
        xb = self.flatten(xb)
        xb = self.fc1(xb)
        xb = self.fc2(xb)
        return xb

    #############################################################  ResNeXt  ############################################################################
    def init_ResNeXt(self, block, num_classes=10):
        """
            This is the initialization of a ResNeXt network. It consists of a convoloutional layer that
            converts the 3-channel image data to a 16-channel data. Then ResNeXt layers containing ResNeXt 
            blocks are added to the network according to the "n_blocks" variable. The output of these 
            layers are passed through an AdaptiveAvgPool layer and then flattened to be passed through 
            two fully connected layers. 
        """

        n_blocks = [3, 4, 6, 3]
        in_channels = 16
        bottleneck_width = 4
        stride = 2
        cardinality = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.ResNeXt_layers = nn.Sequential()
        for i in range(len(n_blocks)):
            out_channels = 2 * in_channels
            self.ResNeXt_layers.add_module(f'ResNeXt_layer_{i+1}', self.make_ResNeXt_layer(block, in_channels, out_channels, n_blocks[i], bottleneck_width, stride, cardinality))
            in_channels *= 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(out_channels, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def make_ResNeXt_layer(self, block, in_channels, out_channels, num_blocks, bottleneck_width=4, stride=1, cardinality=32):
        ResNeXt_layer = nn.Sequential()
        ResNeXt_layer.add_module('ResNext_block_1', block(in_channels, out_channels, bottleneck_width, stride, cardinality))
        for i in range(num_blocks - 1):
            ResNeXt_layer.add_module(f'ResNext_block_{i+2}', block(out_channels, out_channels, bottleneck_width, 1, cardinality))
        
        return ResNeXt_layer

    def ResNeXt_forward(self, xb):
        xb = self.conv1(xb)
        xb = self.ResNeXt_layers(xb)
        xb = self.avgpool(xb)
        xb = self.flatten(xb)
        xb = self.fc1(xb)
        xb = self.fc2(xb)
        xb = self.relu(xb)
        return xb

    ############################################################  DenseNet  ###############################################################################
    def init_DenseNet(self, block, growth_rate=16, reduction=0.5, num_classes=10):
        """
            This is the initialization of a DenseNet network. It consists of a convoloutional layer that
            converts the 3-channel image data to a 16-channel data. Then DenseNet layers containing DenseNet 
            blocks and Transition layers are added to the network according to the "n_blocks" variable. The 
            output of these layers are passed through an AdaptiveAvgPool layer and then flattened to be passed 
            through two fully connected layers. 
        """
        n_blocks = [6, 12, 24, 16]
    
        self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU()
                    )
        
        in_channels = 16
        inner_channels = in_channels
        self.DenseNet_layers = nn.Sequential()
        for i in range(len(n_blocks) - 1):
            self.DenseNet_layers.add_module(f'DenseNet_layer_{i+1}', self.make_DenseNet_layer(block, inner_channels, growth_rate, n_blocks[i]))
            inner_channels += growth_rate * n_blocks[i]

            out_channels = int(reduction * inner_channels)
            self.DenseNet_layers.add_module(f'Transition_layer_{i+1}', self.DenseNet_transition(inner_channels, out_channels))
            inner_channels = out_channels
        
        self.DenseNet_layers.add_module(f'DenseNet_layer_{len(n_blocks)}', self.make_DenseNet_layer(block, inner_channels, growth_rate, n_blocks[-1]))
        inner_channels += growth_rate * n_blocks[-1]
        self.DenseNet_layers.add_module('BatchNorm', nn.BatchNorm2d(inner_channels))
        self.DenseNet_layers.add_module('ReLU', nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(inner_channels, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def make_DenseNet_layer(self, block, in_channels, growth_rate, n_blocks):
        DenseNet_layer = nn.Sequential()
        for i in range(n_blocks):
            DenseNet_layer.add_module(f'DenseNet_block_{i+1}', block(in_channels, growth_rate, bottleneck_width=4))
            in_channels += growth_rate
        return DenseNet_layer
    
    def DenseNet_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def DenseNet_forward(self, xb):
        xb = self.conv1(xb)
        xb = self.DenseNet_layers(xb)
        xb = self.avgpool(xb)
        xb = self.flatten(xb)
        xb = self.fc1(xb)
        xb = self.fc2(xb)
        return xb


    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        
        if self.model_type == 1:  
            out = self.basicnet(xb)
        elif self.model_type == 2:
            out = self.batchnormnet(xb)
        elif self.model_type == 3:
            out = self.ResNet_forward(xb)
        elif self.model_type == 4:
            out = self.ResNeXt_forward(xb)
        elif self.model_type == 5:
            out = self.DenseNet_forward(xb)

        return out
