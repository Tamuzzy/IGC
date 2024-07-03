import torch
import torch.nn as nn

class int_mlp(nn.Module):
    def __init__(self, num_series, hidden, lag, num_int):
        super(int_mlp, self).__init__()
        self.activation = nn.ReLU()
        self.ref_layer = nn.Conv1d(num_series, hidden[0], lag)

        int_net = []
        for i in range(num_int):
            int_layer = nn.Conv1d(num_series, hidden[0], lag)
            int_net.append(int_layer)
        self.int_layer = nn.ModuleList(int_net)

        share_net = []
        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            share_net.append(layer)
        self.share_layer = nn.ModuleList(share_net)

    def forward(self, x, net_id):
        x = x.transpose(2,1)
        ref_x = self.ref_layer(x)
        int_x = self.int_layer[net_id](x)
        x = ref_x + int_x

        for fc in self.share_layer:
            x = self.activation(x)
            x = fc(x)
        return x.transpose(2,1)

class igc(nn.Module):
    def __init__(self, num_series, hidden, lag, num_int):
        super(igc, self).__init__()
        self.p = num_series
        self.lag = lag
        self.interv = num_int

        self.networks = nn.ModuleList([
            int_mlp(self.p, hidden, lag, num_int) for _ in range(self.p)
        ])

    def forward(self, x):
        return torch.cat([network(x) for network in self.networks], dim=2)

    def gc(self, threshold=True, on_lag=True):
        if on_lag:
            granger = [torch.norm(net.ref_layer.weight ,dim=(0,2)) for net in self.networks]
        else:
            granger = [torch.norm(net.ref_layer.weight, dim=0) for net in self.networks]
        granger = torch.stack(granger)
        if threshold:
            return (granger > 0).int()
        else:
            return granger

    def itf(self, threshold=True, on_lag=True):
        int_family = []
        for i in range(self.interv):
            if on_lag:
                target = [torch.norm(net.int_layer[i].weight, dim=(0,2)) for net in self.networks]
            else:
                target = [torch.norm(net.int_layer[i].weight, dim= 0) for net in self.networks]
            target = torch.stack(target)
            int_family.append(target)
        if threshold:
            return (int_family > 0).int()
        else:
            return int_family






