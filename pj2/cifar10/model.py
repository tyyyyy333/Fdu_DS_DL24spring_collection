import torch.nn as nn
import torch
import argparse
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

from torchinfo import summary


class CANet_tiny(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(CANet_tiny, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes


        self.res_0 = AttResBlock(3, 16, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        ), sak=7, sap=3, sas=1)

        self.res_1 = AttResBlock(16, 16, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        ), sak=7, sap=3, sas=1)

        self.res_2 = AttResBlock(16, 32, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32)
        ), sak=5, sap=2, sas=1)

        self.res_3 = AttResBlock(32, 32, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        ), sak=5, sap=2, sas=1)

        self.res_4 = AttResBlock(32, 64, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64)
        ), sak=3, sap=1, sas=1)

        self.res_5 = AttResBlock(64, 64, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64)
        ), sak=3, sap=1, sas=1)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        #self.preavgpool = nn.AdaptiveMaxPool2d((2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(False)
        self.fc1 = nn.Linear(64 * 2 * 2, 27)
        self.fc2 = nn.Linear(27, 10)

    def forward(self, x):
        x = self.res_0(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)

        x = self.avgpool(x)

        x = self.bn(x)

        x = x.view(-1, 256)
        x = self.relu(self.fc1(x))
       # x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        return x

class CANet_light(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(CANet_light, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes


        self.res_0 = AttResBlock(3, 32, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        ), sak=7, sap=3, sas=1)

        self.res_1 = AttResBlock(32, 32, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        ), sak=7, sap=3, sas=1)

        self.res_2 = AttResBlock(32, 64, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64)
        ), sak=5, sap=2, sas=1)

        self.res_3 = AttResBlock(64, 64, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        ), sak=5, sap=2, sas=1)

        self.res_4 = AttResBlock(64, 128, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ), sak=3, sap=1, sas=1)

        self.res_5 = AttResBlock(128, 128, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ), sak=3, sap=1, sas=1)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        #self.preavgpool = nn.AdaptiveMaxPool2d((2, 2))
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(False)
        self.fc1 = nn.Linear(128 * 2 * 2, 54)
        self.fc2 = nn.Linear(54, 10)

    def forward(self, x):
        x = self.res_0(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)

        x = self.avgpool(x)

        x = self.bn(x)

        x = x.view(-1, 512)
        x = self.relu(self.fc1(x))
       # x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        return x


class CANet_pro(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(CANet_pro, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes


        self.res_0 = AttResBlock(3, 128, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        ), sak=7, sap=3, sas=1)

        self.res_1 = AttResBlock(128, 128, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        ), sak=7, sap=3, sas=1)

        self.res_2 = AttResBlock(128, 256, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ), sak=5, sap=2, sas=1)

        self.res_3 = AttResBlock(256, 256, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        ), sak=5, sap=2, sas=1)

        self.res_4 = AttResBlock(256, 512, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(512)
        ), sak=3, sap=1, sas=1)

        self.res_5 = AttResBlock(512, 512, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(512)
        ), sak=3, sap=1, sas=1)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        #self.preavgpool = nn.AdaptiveMaxPool2d((2, 2))
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(False)
        self.fc1 = nn.Linear(512 * 2 * 2, 216)
        self.fc2 = nn.Linear(216, 10)

    def forward(self, x):
        x = self.res_0(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)

        x = self.avgpool(x)

        x = self.bn(x)

        x = x.view(-1, 2048)
        x = self.relu(self.fc1(x))
       # x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        return x

    
    
class AttResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, downsample=None, sek=[3,5,7], sak=3, sap=1, sas=1):
        super(AttResBlock, self).__init__()    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        self.se = SELayer(sek)
        self.sa = SALayer(sak, sap, sas)
        self.unit_transform = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.conv3(x)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        s_out1 = self.sa(out1)
        u_out1 = self.unit_transform(s_out1)
        s_out2 = self.se(out2)
        u_out2 = self.unit_transform(s_out2)

        out = u_out1 * out1 + u_out2 * out2 + residual
        out = self.bn3(out)
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    '''ECA actually'''
    def __init__(self, k_size=[3,5,7]):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size[0], padding=(k_size[0]-1)//2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size[1], padding=(k_size[1]-1)//2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=k_size[2], padding=(k_size[2]-1)//2, bias=False)
        self.w = nn.Parameter(torch.ones(3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y1 = self.w[0] * self.conv1(y)
        y2 = self.w[1] * self.conv2(y)
        y3 = self.w[2] * self.conv3(y)
        y = y1 + y2 + y3
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y
    
class SALayer(nn.Module):
    def __init__(self, kernel_size=7, padding=3, stride=1):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        x1 = x1.expand_as(x)
        return self.sigmoid(x1) * x
    
class CompetAtt(nn.Module):
    def __init__(self, channel, hidden_channel, SAlayer, SELayer):
        super(CompetAtt, self).__init__()
        self.SAlayer = SAlayer
        self.SELayer = SELayer
        self.channel = channel
        self.unit_transform = nn.Sequential(
            nn.Conv2d(channel, hidden_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x):
        x1 = self.SAlayer(x)
        x1 = self.unit_transform(x1)
        x2 = self.SELayer(x)
        x2 = self.unit_transform(x2)
        x = x * x1 + x * x2
        return x

class CANet(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(CANet, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes


        self.res_0 = AttResBlock(3, 64, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        ), sak=7, sap=3, sas=1)

        self.res_1 = AttResBlock(64, 64, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        ), sak=7, sap=3, sas=1)

        self.res_2 = AttResBlock(64, 128, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ), sak=5, sap=2, sas=1)

        self.res_3 = AttResBlock(128, 128, kernel_size=3, stride=1, padding=1, downsample=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        ), sak=5, sap=2, sas=1)

        self.res_4 = AttResBlock(128, 256, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ), sak=3, sap=1, sas=1)

        self.res_5 = AttResBlock(256, 256, kernel_size=3, stride=2, padding=1, downsample=nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ), sak=3, sap=1, sas=1)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        #self.preavgpool = nn.AdaptiveMaxPool2d((2, 2))
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(False)
        self.fc1 = nn.Linear(256 * 2 * 2, 108)
        self.fc2 = nn.Linear(108, 10)

    def forward(self, x):
        x = self.res_0(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)

        x = self.avgpool(x)

        x = self.bn(x)

        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
       # x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        return x

def fetch_res50():
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
 
    return model


def fetch_res34():
    model = models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model  


def fetch_res18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

def fetch_mobilev2():
    model = models.mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    
    return model

def fetch_shufflenet():
    model = models.shufflenet_v2_x0_5(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

def fetch_vgg11():
    model = models.vgg11(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    
    return model

model_map = {
    'CANet_tiny' : CANet_tiny,
    'CANet_pro' : CANet_pro,
    'CANet_light' : CANet_light,
    'CANet' : CANet,
    'res50' : fetch_res50(),
    'res34' : fetch_res34(),
    'res18' : fetch_res18(),
    'mobile_v2' : fetch_mobilev2(),
    'shufflenet' : fetch_shufflenet(),
    'vgg11' : fetch_vgg11()
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Model')
    parser.add_argument('--model', type=str, default='CANet', metavar='N', help='model to summary (default : CANet)')

    args = parser.parse_args()

    if args.model in ['custom_cnn', 'CANet', 'CANet_light', 'CANet_pro', 'CANet_tiny']:
        model = model_map[args.model](input_shape=(3, 32, 32), num_classes=10)
    else:
        model = model_map[args.model]
        
    summary(model, input_size=(1, 3, 32, 32))