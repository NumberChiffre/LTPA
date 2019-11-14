import torch.nn as nn
from LTPA.utils.utilities import *
from LTPA import vgg_config, vgg_in_channels


class Attention(nn.Module):
    def __init__(self, mode: str = 'pc'):
        super(Attention, self).__init__()
        self.mode = mode

    def forward(self, x):
        raise NotImplementedError()

    @staticmethod
    def _make_layers():
        raise NotImplementedError()

    def _get_compatibility_score(self, l, g, level):
        raise NotImplementedError()

    @staticmethod
    def _get_weighted_combination(l, ae):
        raise NotImplementedError()


class VGGAttention(Attention):
    def __init__(self, mode: str = 'pc'):
        """
        :param mode:
        dp for dot product for matching the global and local descriptors
        pc for the use of parametrised compatibility
        """
        super().__init__(mode=mode)

        # features through VGG
        self.features = self._make_layers()

        # right before the 8th, 11th, and 14th layers
        self.l1 = nn.Sequential(*list(self.features)[:22])
        self.l2 = nn.Sequential(*list(self.features)[22:32])
        self.l3 = nn.Sequential(*list(self.features)[32:42])

        # remaining layers before fully-connected
        self.conv_remain = nn.Sequential(*list(self.features)[42:50])

        # 1st fully-connected back to attention estimator layers
        self.fc1 = nn.Linear(512, 512)
        self.ga1 = nn.Linear(512, 256)
        self.ga2 = nn.Linear(512, 512)
        self.ga3 = nn.Linear(512, 512)

        # last fully-connected after weight combinations
        self.fc2 = nn.Linear(256 + 512 + 512, 10)

        if mode == 'pc':
            self.u1 = nn.Conv2d(256, 1, 1)
            self.u2 = nn.Conv2d(512, 1, 1)
            self.u3 = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        conv_remain = self.conv_remain(l3)

        fc1 = self.fc1(conv_remain.view(conv_remain.size(0), -1))
        ga1 = self.ga1(fc1)
        ga2 = self.ga2(fc1)
        ga3 = self.ga3(fc1)

        ae1 = self._get_compatibility_score(l1, ga1, level=1)
        ae2 = self._get_compatibility_score(l2, ga2, level=2)
        ae3 = self._get_compatibility_score(l3, ga3, level=3)

        g1 = self._get_weighted_combination(l1, ae1)
        g2 = self._get_weighted_combination(l2, ae2)
        g3 = self._get_weighted_combination(l3, ae3)

        g = torch.cat((g1, g2, g3), dim=1)
        out = self.fc2(g)

        # need the attention estimators for the image plots
        return [out, ae1, ae2, ae3]

    @staticmethod
    def _make_layers():
        """the making of convolutional layers for any VGG architecture"""
        layers = []
        in_channels = vgg_in_channels
        for x in vgg_config['VGGAttention']:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _get_compatibility_score(self, l, g, level):
        """secret sauce from the paper"""
        if self.mode == 'dp':
            ae = l * g.unsqueeze(2).unsqueeze(3)
            ae = ae.sum(1).unsqueeze(1)
            size = ae.size()
            ae = ae.view(ae.size(0), ae.size(1), -1)
            ae = torch.softmax(ae, dim=2)
            ae = ae.view(size)

        elif self.mode == 'pc':
            ae = l + g.unsqueeze(2).unsqueeze(3)
            if level == 1:
                u = self.u1
            elif level == 2:
                u = self.u2
            elif level == 3:
                u = self.u3
            ae = u(ae)
            size = ae.size()
            ae = ae.view(ae.size(0), ae.size(1), -1)
            ae = F.softmax(ae, dim=2)
            ae = ae.view(size)
        return ae

    @staticmethod
    def _get_weighted_combination(l, ae):
        g = l * ae
        return g.view(g.size(0), g.size(1), -1).sum(2)
