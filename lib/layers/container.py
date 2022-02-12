import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, _logdetgrad_=None):
        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x)
            return x
        else:
            if _logdetgrad_ is None:
                for i in range(len(self.chain)):
                    x, logpx = self.chain[i](x, logpx)
                return x, logpx
            else:
                for i in range(len(self.chain)):
                    x, logpx, _logdetgrad_ = self.chain[i](x, logpx, _logdetgrad_)
                return x, logpx, _logdetgrad_

    def inverse(self, y, logpy=None, _logdetgrad_=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y)
            return y
        else:
            if _logdetgrad_ is None:
                for i in range(len(self.chain) - 1, -1, -1):
                    y, logpy = self.chain[i].inverse(y, logpy)
                return y, logpy
            else:
                 for i in range(len(self.chain) - 1, -1, -1):
                    y, logpy, _logdetgrad_ = self.chain[i].inverse(y, logpy, _logdetgrad_)
                return y, logpy, _logdetgrad_               


class Inverse(nn.Module):

    def __init__(self, flow):
        super(Inverse, self).__init__()
        self.flow = flow

    def forward(self, x, logpx=None, _logdetgrad_=None):
        return self.flow.inverse(x, logpx, _logdetgrad_)

    def inverse(self, y, logpy=None, _logdetgrad_=None):
        return self.flow.forward(y, logpy, _logdetgrad_)
