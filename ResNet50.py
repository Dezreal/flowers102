from torch import nn
from torchvision import models


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # 可以选择冻结卷积层
        # for p in self.parameters():
        #     p.requires_grad = False
        # self.fc = nn.Linear(in_features=2048, out_features=102)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8, inplace=False),
            nn.Linear(512, 102, bias=True)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


resnet50 = models.resnet50(pretrained=True)
net = Net(resnet50)

