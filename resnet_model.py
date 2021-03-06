import torch
import torch.nn as nn


def base_conv(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes,
                     out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=3,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = base_conv(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.GELU()
        self.conv2 = base_conv(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.do = nn.Dropout(p=0.1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        self.do(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(self.pool(out))

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=32, stride=1, padding=16, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=7, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool1d(3, stride=1)
        self.do = nn.Dropout(0.1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        # x = self.layer5(x)
        # print(x.size())
        x = self.avgpool(x)  # 1x1
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.do(x)
        x = self.fc(x)

        return x


def multi_scale(**kwargs):
    model = ResNet(BasicBlock, [4, 3, 3, 4], **kwargs)
    return model
