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

from ema import ExponentialMovingAverage

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
testset = torchvision.datasets.CIFAR100(root='~/.keras/datasets', train=False, download=True, transform=transform_test)

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
    def __init__(self, device, optimizer_factory, batch_size = 128, log_dir = None, use_ema = False, m = 1, k = 0):
        self.device = device
        self.m = m
        self.k = k

        print('==> Building model..')
        self.net = ResNet(BasicBlock, [2,2,2,2], num_classes = 100)
        self.net = self.net.to(device)
        if device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer_factory(list(self.net.parameters()))
        
        #parameter moving average
        self.ema = ExponentialMovingAverage(self.optimizer, 0.9) if use_ema else None

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        
        if log_dir is not None:
            self.writer_train, self.writer_test = SummaryWriter('logs/%s/train' % log_dir), SummaryWriter('logs/%s/test' % log_dir)
            if use_ema:
                self.writer_ema = SummaryWriter('logs/%s/ema' % log_dir)

    # Training
    def _train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return outputs, loss

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        #mtt dataloader repetition
        self.rep_data = []
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs, loss = self._train_step(inputs, targets)

            #parameters moving average
            if self.ema is not None:
                self.ema.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            if self.m <= 1:
                continue
            #mtt repetition
            self.rep_data.append((inputs, targets))
            if len(self.rep_data) == self.k:
                for _ in range(self.m - 1):
                    for inputs, targets in self.rep_data:
                        self._train_step(inputs, targets)
                self.rep_data = []

        return train_loss/(batch_idx+1), 100.*correct/total 

    def test(self, epoch):
        net = self.net

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return test_loss/(batch_idx+1), 100.*correct/total 


    def train_one_epoch(self, epoch):
        train_loss, train_correct = self.train(epoch)
        self.writer_train.add_scalar('loss', train_loss, epoch)
        self.writer_train.add_scalar('accuracy', train_correct, epoch)

        test_loss, test_correct = self.test(epoch)
        self.writer_test.add_scalar('accuracy', test_correct, epoch)
        self.writer_test.add_scalar('loss', test_loss, epoch)
        print('Acc: %.4f%% | Loss: %.3f' % (test_correct, test_loss))
        
        if self.ema is not None:
            self.ema.apply_state()

            ema_loss, ema_correct = self.test(epoch)
            self.writer_ema.add_scalar('accuracy', ema_correct, epoch)
            self.writer_ema.add_scalar('loss', ema_loss, epoch)

            self.ema.restore()
