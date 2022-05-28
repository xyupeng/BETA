import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class BottleNeckMLP(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, num_classes, type1='bn', type2='wn'):
        super(BottleNeckMLP, self).__init__()
        assert type1 in ['bn', 'bn_relu', 'bn_relu_drop']
        # assert type2 in ['wn', 'linear']
        self.fc1 = nn.Linear(feature_dim, bottleneck_dim)
        self.fc1.apply(init_weights)

        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        if type2 == 'wn':
            self.fc2 = weightNorm(nn.Linear(bottleneck_dim, num_classes), name="weight")
            self.fc2.apply(init_weights)
        elif type2 == 'linear':
            self.fc2 = nn.Linear(bottleneck_dim, num_classes)
            self.fc2.apply(init_weights)
        else:
            self.fc2 = nn.Linear(bottleneck_dim, num_classes, bias=False)
            nn.init.xavier_normal_(self.fc2.weight)
        self.type1 = type1
        self.type2 = type2

    def forward(self, x):
        x = self.fc1(x)
        if 'bn' in self.type1:
            x = self.bn(x)
        if 'relu' in self.type1:
            x = self.relu(x)
        if 'drop' in self.type1:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

