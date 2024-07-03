import torch

def regularize(network, glam, elam):
    w_ref = network.ref_layer.weight
    w_int = network.int_layer

    int_norm = sum([torch.sum(torch.norm(net.weight,dim=(0,2))) for net in w_int])

    for net in w_int:
        w_ref = torch.cat((w_ref, net.weight), dim=2)
    ref_norm = torch.sum(torch.norm(w_ref, dim=(0, 2)))

    return elam * int_norm + glam * ref_norm

def ridge_regularize(network, lam):
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.share_layer])

def prox_update(network, elam, glam, lr):
    w_ref = network.ref_layer.weight
    w_int = network.int_layer

    for net in w_int:
        norm = torch.norm(net.weight, dim=(0,2), keepdim=True)
        net.weight.data = ((net.weight / torch.clamp(norm, min=(lr * elam))) * torch.clamp(norm - lr * elam, min=0.0))

    W = torch.cat((w_ref, w_int[0].weight), dim=2)

    for index, net in enumerate(w_int):
        if index != 0:
            W = torch.cat((W, w_int[index].weight), dim=2)
    norm = torch.norm(W, dim=(0,2), keepdim=True)

    w_ref.data = ((w_ref / torch.clamp(norm, min=(lr * glam))) * torch.clamp(norm - lr * glam, min=0.0))
    for net in w_int:
        net.weight.data = ((net.weight / torch.clamp(norm, min=(lr * glam))) * torch.clamp(norm - lr * glam, min=0.0))

def save_para(model, best_model):
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params





