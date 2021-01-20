
import os

import torch
from torchvision import transforms, datasets

# device = torch.device("cuda:0")
# 对三种数据集进行不同预处理，对训练数据进行加强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 数据目录
data_dir = "/Users/Konyaka/Downloads/flowers102/dataset_split/"

# 获取三个数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'valid', 'test']}
traindataset = image_datasets['train']
validdataset = image_datasets['valid']
testdataset = image_datasets['test']

batch_size = 8
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True) for x in ['train', 'valid', 'test']}
traindataloader = dataloaders['train']
validdataloader = dataloaders['valid']
testdataloader = dataloaders['test']

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
print(dataset_sizes)
