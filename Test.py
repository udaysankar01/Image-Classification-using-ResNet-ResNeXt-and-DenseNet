#!/usr/bin/env python3

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


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.notebook import tqdm
import torch
from Network.Network import CIFAR10Model
from torchvision.datasets import CIFAR10
from Misc.MiscUtils import *
from Misc.DataUtils import *
from ptflops import get_model_complexity_info
import re

# Don't generate pyc codes
sys.dont_write_bytecode = True

# setting the device as 'cuda'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def get_inference_time(device, modelNumber):
    """
        Returns the inference time of the passed in model.
    """
    with torch.cuda.device('cuda'):
        model = CIFAR10Model(modelNumber)
        macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
        print(macs)
        model_flops = float(re.findall(r'\d+.\d+', macs)[0]) / 2

        print(f'Inference time: {model_flops * 1e6 / (16.58 * 1e6):.3f} microseconds')

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(LabelsTrue)))
    disp.plot()
    plt.show()

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelNumber):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """

    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = CIFAR10Model(ModelNumber) 
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model = model.to(device)
    # print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    print('Number of parameters in this model are %d ' % sum(p.numel() for p in model.parameters()))
    
    OutSaveT = open(LabelsPathPred, 'w')

    model.eval()

    for count in tqdm(range(len(TestSet))): 
        Img, Label = TestSet[count]
        Img, ImgOrg = ReadImages(Img)
        # PredT = torch.argmax(model(Img)).item()
        model_input = torch.tensor(Img).to(device)
        PredT = torch.argmax(model(model_input)).item()

        OutSaveT.write(str(PredT)+'\n')
    OutSaveT.close()

       
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='Checkpoints/BasicNet/14model.ckpt', help='Path to load latest model from, Default:Checkpoints/BasicNet/14model.ckpt')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    TestSet = CIFAR10(root='data/', train=False, download=True, transform=ToTensor())
    TrainSet = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())

    model_dict = {'BasicNet': 1, 'BatchNormNet': 2, 'ResNet': 3, 'ResNeXt': 4, 'Densenet': 5}
    ModelNumber = 1 # default
    for key in model_dict.keys():
        if key in ModelPath:
            ModelNumber = model_dict[key]
            break

    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    LabelsPathPredTrain = './TxtFiles/PredTrainOut.txt'
    LabelsPathTrain = "./TxtFiles/LabelsTrain.txt"   

    TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelNumber)

    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred) 

    # Plot Training Confusion Matrix
    TestOperation(ImageSize, ModelPath, TrainSet, LabelsPathPredTrain, ModelNumber)
    LabelsTrue, LabelsPred = ReadLabels(LabelsPathTrain, LabelsPathPredTrain)
    ConfusionMatrix(LabelsTrue, LabelsPred) 

    get_inference_time(device, ModelNumber)
     
if __name__ == '__main__':
    main()
 
