import torch as th
import th.nn as nn
import math
from BBBLinear import BBB_Linear
from BBBConv import BBB_Conv2d
from Module_Wrapper import FlattenLayer, ModuleWrapper

class BBBAlexNet(ModuleWrapper):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BBBAlexNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        BBBLinear = BBB_Linear
        BBBConv2d = BBB_Conv2d
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 64, 11, stride=4, padding=5, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(64, 192, 5, padding=2, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(192, 384, 3, padding=1, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.conv4 = BBBConv2d(384, 256, 3, padding=1, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.conv5 = BBBConv2d(256, 128, 3, padding=1, bias=True, priors=self.priors)
        self.act5 = self.act()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(1 * 1 * 128)
        self.classifier = BBBLinear(1 * 1 * 128, outputs, bias=True, priors=self.priors)
