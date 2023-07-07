import torch.nn as nn
import torch.nn.functional as F
import torch


def get_model(nw_arch: str,                         # n/w architecture
              num_classes: int,                     # dimension of linear layer
              proj_dim=512):                        # projection dimension
    """ wrapper to return appropriate model class """
    model = Model(nw_arch=nw_arch,
                  num_classes=num_classes,
                  proj_dim=proj_dim)

    print('------------------')
    print('Loading Model: {}'.format(nw_arch))
    print('------------------')
    print(model)
    print('------------------')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num of Params = {}'.format(num_params))
    print('------------------')

    return model, num_params


class Model(nn.Module):
    def __init__(self,
                 nw_arch: str,
                 num_classes: int,
                 proj_dim: int):

        self.nw_arch = nw_arch
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        super(Model, self).__init__()

        if nw_arch == 'cifar_resnet18':
            self.backbone = CIFARResNet(BasicBlock, [2, 2, 2, 2])
            self.feat_dim = 512

        else:
            raise NotImplementedError

        self.projector = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim, bias=False),
                                       nn.BatchNorm1d(self.feat_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.feat_dim, self.proj_dim, bias=True),
                                       nn.BatchNorm1d(self.proj_dim))

        self.linear_head = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, feature, mode: str):
        """
        :param feature:
        :param mode:
        :return:
        """
        softmax = nn.LogSoftmax(dim=1)

        if mode == 'linear_probe':
            h = F.relu(feature)
            h = self.linear_head(h)
            return softmax(h)

        else:
            r = self.backbone(feature)
            h = torch.flatten(r, start_dim=1)

            if mode == 'encode':
                return self.projector(h)

            elif mode == 'finetune':
                h = F.relu(h)
                h = self.linear_head(h)
                return softmax(h)

            elif mode == 'feat_ext':
                return h

            else:
                raise NotImplementedError


# ----- Define Backbones ------ #
def conv3x3(in_planes, out_planes, stride=1):
    """
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(CIFARResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out
