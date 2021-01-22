import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from data import validdataloader, dataset_sizes, testdataloader, traindataloader, batch_size
from ResNet152 import net

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
device = torch.device("cpu")


def test(model, mode):
    print('-' * 10)
    total_loss = 0.0
    total_corrects = 0
    model = model.to(device)
    if mode == 'valid':
        loader = validdataloader
    elif mode == 'test':
        loader = testdataloader
    else:
        loader = None
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        outputs = loss(outputs, labels)
        total_loss += outputs.item()
        total_corrects += torch.sum(preds == labels)
    epoch_loss = total_loss * batch_size / dataset_sizes[mode]
    epoch_acc = total_corrects.double() / dataset_sizes[mode]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        mode, epoch_loss, epoch_acc))
    print('-' * 10)
    print()


def valid_model(model):
    test(model, 'valid')


def test_model(model):
    test(model, 'test')


def train_model(model, optimizer, num_epochs=5):
    since = time.time()
    x = np.array([])
    y_lss = np.array([])
    y_acc = np.array([])
    model = model.to(device)
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        total_loss = 0.0
        total_corrects = 0
        for i, (inputs, labels) in enumerate(traindataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            
            outputs = loss(outputs, labels)
            outputs.backward()
            optimizer.step()
            total_loss += outputs.item()
            total_corrects += torch.sum(preds == labels)
            if (i + 1) % 200 == 0:
                print("batch: %d / %d Loss: %.4f" % ((i + 1), len(traindataloader), outputs.item()))
        epoch_loss = total_loss * batch_size / dataset_sizes['train']
        epoch_acc = total_corrects.double().item() / dataset_sizes['train']
        x = np.append(x, epoch + 1)
        y_acc = np.append(y_acc, epoch_acc)
        y_lss = np.append(y_lss, epoch_loss)
        plt.plot(x, y_acc, '-og')
        plt.plot(x, y_lss, '-or')
        plt.pause(0.01)
        best_acc = max(best_acc, epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))
        print()
        if (epoch + 1) % 2 == 0:
            valid_model(model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


if __name__ == "__main__":

    # load and train
    model = torch.load('model20.pth', map_location=torch.device('cpu'))
    epochs = 0
    model = train_model(model, optimizer, epochs)
    test_model(model)
    plt.show()

    # train only
    # epochs = 20
    # model = train_model(net, optimizer, epochs)
    # test_model(model)
    # plt.show()

    # save
    # torch.save(model, 'model.pth')
