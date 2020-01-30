import os
import torch
import math
from torch import nn
import torch.nn.functional as F
import src.segmentation_models_pytorch as smp
from torchvision import models
from efficientnet_pytorch import EfficientNet
from src.efficientunet import *

## Swish activation function ##
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
#         self.inplace = inplace
    def forward(self, x, beta=1.12):
        return x * torch.sigmoid(beta * x)
def convert_relu_to_swish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, swish())
        else:
            convert_relu_to_swish(child)
            
def get_torchvision_model(net_type, is_trained, num_classes, loss):
    """ Get torchvision model

    Parameters
    ----------
    net_type: str
        deep network type
    is_trained: boolean
        use pretrained ImageNet
    num_classes: int
        number of classes

    Returns
    -------
    nn.Module
        model based on net_type
    """
    if net_type.startswith("Se_resnext"):
        return Se_resnext(net_type, is_trained, num_classes, loss)
    elif net_type.startswith("Se_resnet"):
        return Se_resnet(net_type, is_trained, num_classes, loss)
    elif net_type.startswith("Resnet"):
        return Resnet(net_type, is_trained, num_classes, loss)
    else:
        return CustomEfficientNet(net_type, is_trained, num_classes, loss)

class CustomEfficientNet(nn.Module):
    """
    efficientnetb0: net_type
    """
    def __init__(self, net_type, is_trained, num_classes, loss):
        super().__init__()
        if net_type.endswith("b0"):
            self.net = get_efficientunet_b0(out_channels=num_classes, concat_input=True, pretrained = is_trained)
        elif net_type.endswith("b1"):
            self.net = get_efficientunet_b1(out_channels=num_classes, concat_input=True, pretrained = is_trained)
        elif net_type.endswith("b2"):
            self.net = get_efficientunet_b2(out_channels=num_classes, concat_input=True, pretrained = is_trained)
        elif net_type.endswith("b3"):
            self.net = get_efficientunet_b3(out_channels=num_classes, concat_input=True, pretrained =True)
        elif net_type.endswith("b4"):
            self.net = get_efficientunet_b4(out_channels=num_classes, concat_input=True, pretrained = is_trained)
        elif net_type.endswith("b5"):
            self.net = get_efficientunet_b5(out_channels=num_classes, concat_input=True, pretrained = is_trained)
        elif net_type.endswith("b6"):
            self.net = get_efficientunet_b6(out_channels=num_classes, concat_input=True, pretrained = is_trained)
        else:
            self.net = get_efficientunet_b7(out_channels=num_classes, concat_input=True, pretrained = is_trained)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.net(x)
        if not self.training:
             x = self.sigmoid(x)
        return x
    
class Se_resnext(nn.Module):
    """
    Seresnext50:  net_type = "Se_resnext50"
    Seresnext101: net_type = "Se_resnext101"

    """
    def __init__(self, net_type, is_trained, num_classes, loss):
        super().__init__()
        if net_type.endswith("50"):
            if is_trained:
                self.net = smp.Unet("se_resnext50_32x4d", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("se_resnext50_32x4d",  classes = num_classes)
        else:
            if is_trained:
                self.net = smp.Unet("se_resnext101_32x4d", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("se_resnext101_32x4d",  classes = num_classes)
        self.sigmoid = nn.Sigmoid()
        
        #convert ReLU to swish:
        convert_relu_to_swish(self.net)
    def forward(self, x):
        x = self.net(x)
        if not self.training:
             x = self.sigmoid(x)
        return x

class Se_resnet(nn.Module):

    """
    Seresnet50:  net_type = "Se_resnet50"
    Seresnet101: net_type = "Se_resnet101"
    Seresnet152: net_type = "Se_resnet152"

    """

    def __init__(self, net_type, is_trained, num_classes, loss):
        super().__init__()
        if net_type.endswith("50"):
            if is_trained:
                self.net = smp.Unet("se_resnet50", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("se_resnet50",  classes = num_classes)
        elif net_type.endswith("101"):
            if is_trained:
                self.net = smp.Unet("se_resnet101", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("se_resnet101",  classes = num_classes)
        else:
            if is_trained:
                self.net = smp.Unet("se_resnet152", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("se_resnet152",  classes = num_classes) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.net(x)
        if not self.training:
             x = self.sigmoid(x)
        return x
        
class Resnet(nn.Module):
    """
    Resnet18: net_type = "Resnet18"
    Resnet34: net_type = "Resnet34"
    Resnet50: net_type = "Resnet50"
    Resnet101: net_type = "Resnet101"
    Resnet152: net_type = "Resnet152"

    """

    def __init__(self, net_type, is_trained, num_classes, loss):
        super().__init__()
        if net_type.endswith("18"):
            if is_trained:
                self.net = smp.Unet("resnet18", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("resnet18",  classes = num_classes)
        elif net_type.endswith("34"):
            if is_trained:
                self.net = smp.Unet("resnet34", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("resnet34",  classes = num_classes)
        
        elif net_type.endswith("50"):
            if is_trained:
                self.net = smp.Unet("resnet50", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("resnet50",  classes = num_classes)
        elif net_type.endswith("101"):
            if is_trained:
                self.net = smp.Unet("resnet101", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("resnet101",  classes = num_classes)
        else:
            if is_trained:
                self.net = smp.Unet("resnet152", encoder_weights='imagenet', classes = num_classes)
            else:
                self.net = smp.Unet("resnet152",  classes = num_classes) 
      
        self.sigmoid = nn.Sigmoid()
        
        #convert ReLU to swish:
        convert_relu_to_swish(self.net)

    def forward(self, x):
        x = self.net(x)
        if not self.training:
             x = self.sigmoid(x)
        return x
