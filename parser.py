import argparse
import optimizers as opts

optimizer_choices = ['base', 'lookahead', 'mtt', 'qhm']
base_choices = ['sgd', 'mom', 'mom2', 'mom3']


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
            values.append(0)
        dest = getattr(namespace, self.dest)
        for v in values:
            dest.append((option_strings[2:], v))
for base in base_choices:
    parser.add_argument('--%s'%base, action=Mtt_Action, dest='mtt_bases', type=float, nargs='*')
#mtt_dataload
parser.add_argument('--mtt_data_repeat', action='store_true')
#batch_size
parser.add_argument('--b', default=128, type=int)
#lr_milestone
parser.add_argument('--ms', type=int, nargs='+')
parser.add_argument('--total_epoch', type=int, default=200)
#parameter moving average
parser.add_argument('--ema', action='store_true')
#gpu
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

#lr_milestone default
if args.ms is None:
    args.ms = [60, 120, 160, 180]

#using optimizer factory
lr_wd_dict = {'lr':args.lr, 'weight_decay':args.weight_decay}
def get_optimizer_factory():
    if args.mode == 'base':
        return opts.get_base_optimizer_factory(args.opt, **lr_wd_dict)
    elif args.mode == 'lookahead':
        _base =  opts.get_base_optimizer_factory(args.opt, **lr_wd_dict)
        return opts.get_lookahead_factory(_base, k=args.k)
    elif args.mode == 'mtt':
        #mtt optimizer factory
        #check --m & --a logic
        if args.m != 0:
            if args.mtt_bases is None:
                args.mtt_bases = [('mom', 0)]
            assert(len(args.mtt_bases) == 1)
            _base = args.mtt_bases[0]
            args.mtt_bases = [_base] * args.m
            _v_s = [1/args.m] * args.m
        else:
            #mode: append opt&ratio
            assert(len(args.mtt_bases) > 0)
            _v_s = [_r for _, _r in args.mtt_bases]

        _bases = []
        for _base, _ratio in args.mtt_bases:
            _bases.append(opts.get_base_optimizer_factory(_base, **lr_wd_dict))
        return opts.get_mtt_factory(_bases, v_s=_v_s, k=args.k)
    else:
        _sgd = opts.get_base_optimizer_factory('sgd', **lr_wd_dict)
        _mom3 = opts.get_base_optimizer_factory('mom3', **lr_wd_dict)
        return opts.get_mtt_factory((_sgd, _mom3), v_s=(0.3, 0.7), k=1)
"""
Naming:
    _mode:'mode'<_opt_ps>_lr:'lr'_dk:'dk'_wd:'wd'_b:'bs'

    <_opt_ps>:
        normal training: qhm:
            ''
        lookahead:
            _k:'k'_'opt'
        mtt:
            _m:'m'_k:'k'_<diff|same>
"""
def get_model_name():
    _mode = 'mode:%s' % args.mode
    if args.mode == 'base':
        _mode = 'mode:%s' % args.opt

    _opt_ps = ''
    if args.mode == 'lookahead':
        _opt_ps = '_k:%d_%s' % (args.k, args.opt)
    elif args.mode == 'mtt':
        _opt_ps = '_m:%d_k:%d_%s%s' % (len(args.mtt_bases), args.k, 'same' if args.m != 0 else 'diff', '_repeat' if args.mtt_data_repeat else '')

    _lr = 'lr:%.0e' % args.lr
    _dk = 'dk:%.0e' % args.dk
    _wd = 'wd:%.0e' % args.weight_decay
    _b = 'b:%d' % args.b
    _ema = 'ema' if args.ema else ''

    return '_%s%s_%s_%s_%s_%s_%s' % (_mode, _opt_ps, _lr, _dk, _wd, _b, _ema)

if __name__ == '__main__':
    print(args)
    get_optimizer_factory()
