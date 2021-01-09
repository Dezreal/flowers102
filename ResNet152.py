from torch import nn
from torchvision import models


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # 可以选择冻结卷积层
        # for p in self.parameters():
        #     p.requires_grad = False
        self.fc = nn.Linear(in_features=2048, out_features=102)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


resnet152 = models.resnet152(pretrained=True)
net = Net(resnet152)

