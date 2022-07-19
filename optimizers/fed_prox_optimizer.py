from torch.optim import Optimizer


class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr, lamda=lamda, mu=mu)
        super(FedProxOptimizer, self).__init__(params, defaults)

    def step(self, vstar, closure=None):
        loss=None
        if closure is not None:
            loss=closure
        for group in self.param_groups:
            for p, pstar in zip(group['params'], vstar):
                # w <=== w - lr * ( w'  + lambda * (w - w* ) + mu * w )
                p.data=p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - pstar.data.clone()) + group['mu'] * p.data)
        return group['params'], loss
