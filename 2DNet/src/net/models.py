from torch import autograd
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
#from efficientnet_pytorch import EfficientNet
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import sys

# class AdaptiveConcatPool2d(nn.Module): # 사용 X
#     def __init__(self, sz=None):
#         super().__init__()
#         sz = sz or (1,1)
#         self.ap = nn.AdaptiveAvgPool2d(sz)
#         self.mp = nn.AdaptiveMaxPool2d(sz)

#     def forward(self, x):
#         return torch.cat([self.ap(x), self.mp(x)], 1)

# def l2_norm(input, axis=1):
#     norm = torch.norm(input,2, axis, True)
#     output = torch.div(input, norm)
#     return output

# class DenseNet169_change_avg(nn.Module):
#     def __init__(self):
#         super(DenseNet169_change_avg, self).__init__()
#         self.densenet169 = torchvision.models.densenet169(pretrained=True).features
#         self.avgpool = nn.AdaptiveAvgPool2d(1)  
#         self.relu = nn.ReLU()
#         self.mlp = nn.Linear(1664, 6)
#         self.sigmoid = nn.Sigmoid()   

#     def forward(self, x):
#         x = self.densenet169(x)      
#         x = self.relu(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.mlp(x)

#         return x

# class DenseNet121_change_avg(nn.Module): #default model
#     def __init__(self):
#         super(DenseNet121_change_avg, self).__init__()
#         self.densenet121 = torchvision.models.densenet121(pretrained=True).features
    
#         # input channel 바꾸기
#         #self.densenet121 = torchvision.models.densenet121(pretrained=True)

#         # new_classifier = nn.Sequential(*list(self.densenet121.classifier.children())[:-1])
#         # self.densenet121.classifier = new_classifier

#         # prev_w = self.densenet121.features.conv0.weight
#         # self.densenet121.features.conv0 = nn.Conv2d(48, 64, kernel_size=(7, 7), stride=(2, 2), padding=(4, 4), bias=False)   
#         # self.densenet121.features.conv0.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64, 45, 7, 7)), dim=1))
        
#         #self.densenet121.classifier = nn.Linear(4096, 6) # 1024,1000


#         self.avgpool = nn.AdaptiveAvgPool2d(1)  
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(1024, 6) 
#         #self.fc1 = nn.Linear(4096, 6)  
#         self.sigmoid = nn.Sigmoid()   

#     def forward(self, x):
#         x = self.densenet121(x)  #4096,1000
#         x = self.relu(x)
#         x = self.avgpool(x) #1000,1
#         x = x.view(x.size(0), -1) 
#         x = self.fc1(x)

        
#         return x

class se_resnext101_32x4d(nn.Module):
    def __init__(self):
        super(se_resnext101_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        #self.model_ft = torch.load('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/src/data_test/se_resnext101_32x4d/model_epoch_best_0.pth')

        prev_w = self.model_ft.layer0.conv1.weight
        self.model_ft.layer0.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.model_ft.layer0.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64, 9, 7, 7)), dim=1))
        #print(self.model_ft.layer0.conv1.weight.shape)
        
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 1, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x



######################### ECA_module ##############################
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECABottleneck(nn.Module): #101 
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size) #efficient channel attention module
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out) #마지막 eca module

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    #ECA_block, [3, 4, 23, 3], num_classes=6, k_size=3
    def __init__(self, block, layers, num_classes=1, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # prev_w = self.model.layer0.conv1.weight 
        # self.model_ft.layer0.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64, 9, 7, 7)), dim=1))

        self.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)   
        self.sigmoid = nn.Sigmoid()  
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 4, 23, 3], num_classes=1, k_size=k_size)
    pretrained_model = torch.load('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/src/data_test/ECANet/model/eca_resnet101_k3357.pth.tar')
    state = model.state_dict()  
    for key in state.keys():
        if key in pretrained_model.keys():
            state[key] = pretrained_model[key]
            model.load_state_dict(state)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model







