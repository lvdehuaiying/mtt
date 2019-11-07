'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from pytorch_cifar.models import *
from pytorch_cifar.utils import progress_bar

from torch.utils.tensorboard import SummaryWriter

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='~/.keras/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='~/.keras/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()

class cifar_trainer():
    def __init__(self, device, optimizer_factory, log_dir = None):
        self.device = device

        print('==> Building model..')
        self.net = ResNet(BasicBlock, [2,2,2,2], num_classes = 100)
        self.net = self.net.to(device)
        if device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer_factory(list(self.net.parameters()))

        if log_dir is not None:
            self.writer_train, self.writer_test = SummaryWriter('logs/%s/train' % log_dir), SummaryWriter('logs/%s/test' % log_dir)

    # Training
    def train(self, epoch, optimizer):
        print('\nEpoch: %d' % epoch)
        
        net = self.net
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return train_loss/(batch_idx+1), 100.*correct/total 

    def test(self, epoch, optimizer):
        net = self.net

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return test_loss/(batch_idx+1), 100.*correct/total 


    def train_one_epoch(self, epoch):
        optimizer = self.optimizer
        train_loss, train_correct = self.train(epoch,optimizer)
        self.writer_train.add_scalar('loss', train_loss, epoch)
        self.writer_train.add_scalar('accuracy', train_correct, epoch)

        test_loss, test_correct = self.test(epoch,optimizer)
        self.writer_test.add_scalar('accuracy', test_correct, epoch)
        self.writer_test.add_scalar('loss', test_loss, epoch)
        print('Acc: %.4f%% | Loss: %.3f' % (test_correct, test_loss))

