import pytorch_cifar.main as pc
import torch.optim as optim
import lookahead
import mtt

import os
from parser import args, get_optimizer_factory, get_model_name

device = 'cuda' if args.gpu else 'cpu'

optimizer_factory = get_optimizer_factory()

#log dir
if not os.path.isdir('logs'):
    nums = 0
else:
    nums = len(os.listdir('logs'))
log_name = 'cifar100%s_%d' % (get_model_name(), nums)

#trainer initializer
trainer = pc.cifar_trainer(device, optimizer_factory, args.b, log_name)
#learning rate scheduler
scheduler = optim.lr_scheduler.MultiStepLR(trainer.optimizer, milestones=args.ms, gamma=args.dk)

if __name__ == '__main__':
    for epoch in range(200):
        trainer.train_one_epoch(epoch)
        scheduler.step()
