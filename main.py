import pytorch_cifar.main as pc
import torch.optim as optim
import lookahead
import mtt
import optimizers as opts

import os
import argparse

optimizer_choices = ['base', 'lookahead', 'mtt', 'qhm']
base_choices = ['sgd', 'mom']

lr_milestones = [60, 120, 160, 180]

"""
Usage:
    python main.py [base|lookahead|mtt|qhm]
    normal training:
        --opt mom --lr 0.1 --dk 0.2 --weight_decay 5e-4 --gpu
    lookahead training:
        --opt mom --k 5
    mtt training:
        --m 2 --mom --k 5
        --a --sgd 0.3 --mom 0.5 --sgd 0.2 --k 5
    qhm training:
        --qhm
"""

parser = argparse.ArgumentParser()
#traning mode
parser.add_argument('mode', nargs='?', default='base', choices=optimizer_choices)
#hyper-parameters setting
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dk', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=5e-4, type=float)
#base optimizers
parser.add_argument('--opt', default='mom', choices = base_choices)
#lookahead
parser.add_argument('--k', default=5, type=int)
#mtt
parser.add_argument('--m', default=2, type=int)
parser.add_argument('--a', action='store_const', dest='m', const=0)
class Mtt_Action(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(Mtt_Action, self).__init__(*args, **kwargs)
    def __call__(self, parser, namespace, values, option_strings=None):
        if not getattr(namespace, self.dest):
            setattr(namespace, self.dest, list())
        if len(values) == 0:
            values.append(-1)
        dest = getattr(namespace, self.dest)
        for v in values:
            dest.append((option_strings[2:], v))
for base in base_choices:
    parser.add_argument('--%s'%base, action=Mtt_Action, dest='mtt_bases', type=float, nargs='*')
#gpu
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

device = 'cuda' if args.gpu else 'cpu'

#using optimizer factory
lr_wd_dict = {'lr':args.lr, 'weight_decay':args.weight_decay}
def get_optimizer_factory():
    if args.mode == 'base':
        return opts.get_base_optimizer_factory(args.opt, **lr_wd_dict)
    elif args.mode == 'lookahead':
        _base =  opts.get_base_optimizer_factory(args.opt, **lr_wd_dict)
        return opts.get_lookahead_factory(_base, k=args.k)
    else:
        #mtt optimizer factory
        #check --m & --a logic
        if args.m != 0:
            if args.mtt_bases is None:
                args.mtt_bases = [('mom', 0)]
            assert(len(args.mtt_bases) == 1)
            _base = args.mtt_bases[0]
            args.mtt_bases = [_base for _ in range(args.m)]
        else:
            #mode: append opt&ratio
            assert(len(args.mtt_bases) > 0)
            assert(sum([_r for _, _r in args.mtt_bases]) == 1.0)

        _bases = []
        for _base, _ratio in args.mtt_bases:
            _bases.append(opts.get_base_optimizer_factory(_base, **lr_wd_dict))
        return opts.get_mtt_factory(_bases, k=args.k)
optimizer_factory = get_optimizer_factory()

#log dir
if not os.path.isdir('logs'):
    nums = 0
else:
    nums = len(os.listdir('logs'))
"""
Naming:
    _mode:'mode'<_opt_ps>_lr:'lr'_dk:'dk'_wd:'wd'

    <_opt_ps>:
        normal training: qhm:
            ''
        lookahead:
            _k:'k'_'opt'
        mtt:
            _m:'m'_k:'k'_<diff|same>
"""
def model_name():
    _mode = 'mode:%s' % args.mode
    if args.mode == 'base':
        _mode = 'mode:%s' % args.opt

    _opt_ps = ''
    if args.mode == 'lookahead':
        _opt_ps = '_k:%d_%s' % (args.k, args.opt)
    elif args.mode == 'mtt':
        _opt_ps = '_m:%d_k:%d_%s' % (len(args.mtt_bases), args.k, 'same' if args.m != 0 else 'diff')

    _lr = 'lr:%.0e' % args.lr
    _dk = 'dk:%.0e' % args.dk
    _wd = 'wd:%.0e' % args.weight_decay

    return '_%s%s_%s_%s_%s' % (_mode, _opt_ps, _lr, _dk, _wd)
log_name = 'cifar100%s_%d' % (model_name(), nums)

#trainer initializer
trainer = pc.cifar_trainer(device, optimizer_factory, log_name)
#learning rate scheduler
scheduler = optim.lr_scheduler.MultiStepLR(trainer.optimizer, milestones=lr_milestones, gamma=args.dk)

if __name__ == '__main__':
    for epoch in range(200):
        trainer.train_one_epoch(epoch)
        scheduler.step()
