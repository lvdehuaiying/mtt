import pytorch_cifar.main as pc
import lookahead
import mtt

def model_opt():
    if pc.args.lookahead:
        return lookahead.Lookahead(pc.optimizer, k=5, alpha=0.5)
    if pc.args.mtt:
        return mtt.Multipath(pc.optimizer, k=5, m=4, alpha=0.5)
    return pc.optimizer

if __name__ == '__main__':
    for epoch in range(pc.start_epoch, pc.start_epoch+200):
        pc.train_one_epoch(epoch, model_opt())
