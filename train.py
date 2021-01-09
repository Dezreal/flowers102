import time

import torch
from torch import nn

from data import validdataloader, dataset_sizes, testdataloader, traindataloader
from ResNet152 import net

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=0.0001,momentum=0.9)
device = torch.device("cpu")



def valid_model(model, criterion):
    best_acc = 0.0
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0
    model = model.to(device)
    for inputs, labels in validdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels)
    epoch_loss = running_loss / dataset_sizes['valid']
    print(running_corrects.double())
    epoch_acc = running_corrects.double() / dataset_sizes['valid']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'valid', epoch_loss, epoch_acc))
    print('-' * 10)
    print()


def test_model(model, criterion):
    best_acc = 0.0
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0
    model = model.to(device)
    for inputs, labels in testdataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels)
    epoch_loss = running_loss / dataset_sizes['test']
    print(running_corrects.double())
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', epoch_loss, epoch_acc))
    print('-' * 10)
    print()


def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        if (epoch + 1) % 5 == 0:
            test_model(model, criterion)
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        running_loss = 0.0
        running_corrects = 0
        model = model.to(device)
        for i, (inputs, labels) in enumerate(traindataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(i)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)
        epoch_loss = running_loss / dataset_sizes['train']
        print(dataset_sizes['train'])
        print(running_corrects.double())
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        best_acc = max(best_acc, epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


if __name__ == "__main__":
    epochs = 1
    model = train_model(net, criterion, optimizer, epochs)

    valid_model(model, criterion)

    torch.save(model, 'model.pkl')
