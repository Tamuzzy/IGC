import torch.nn as nn
import numpy as np
import regularizer

def train(model, X, lr, max_iter, elam=0.0, glam=0.0, lam_ridge=0.0, lookback=5, check_every=20, verbose=1):

    lag = model.lag
    p = X[0].shape[-1]
    mse = nn.MSELoss(reduction="mean")
    loss_list = []

    best_it = None
    best_loss = np.inf
    best_model = None

    loss = 0
    for index, x in enumerate(X):
        loss += sum([mse(model.networks[i](x[:,:-1], net_id=index), x[:,lag:,i:i+1]) for i in range(p)])

    ridge = sum([regularizer.ridge_regularize(net,lam_ridge) for net in model.networks])
    smooth = loss + ridge

    for it in range(max_iter):
        smooth.backward()

        for param in model.parameters():
            param.data = param - lr * param.grad

        if glam > 0:
            for net in model.networks:
                regularizer.prox_update(net,elam,glam,lr)

        model.zero_grad()

        loss = 0
        for index, x in enumerate(X):
            loss += sum([mse(model.networks[i](x[:,:-1], net_id=index), x[:,lag:,i:i+1]) for i in range(p)])
        ridge = sum([regularizer.ridge_regularize(net,lam_ridge) for net in model.networks])
        smooth = loss + ridge

        if (it + 1) % check_every == 0:
            nonsmooth = sum([regularizer.regularize(net, glam, elam) for net in model.networks])
            mean_loss = (smooth + nonsmooth) / p
            loss_list.append(mean_loss.detach())

            if verbose > 0:
                print("Iteration:", it, "Loss", mean_loss)
                print(model.gc(threshold=False))
                #print(model.itf(threshold=False))
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print("Early Stopping")
                break