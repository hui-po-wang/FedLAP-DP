import copy
import numpy as np

import torch.optim as optim

from scipy.spatial import distance

from utils.utils import compute_model_diff

def get_global_opt(args, net):
    if args.optimization == 'fedadam':
        return optim.Adam(params=net.parameters(), lr=args.lr_net)
    elif args.optimization == 'fedlap':
        return optim.SGD(params=net.parameters(), lr=args.lr_net)
    else:
        return optim.SGD(params=net.parameters(), lr=args.lr_net)

def accumulate_grads(model):
    out = []
    for p in model.parameters():
        out.append(p.grad.detach().view(-1).cpu().numpy())
    return np.concatenate(out, axis=0)

def measure_eps(args, net, net_copy, metric, syn_loader, test_loader, real_loader=None, record_grad=False):
    # metric: measure utility of the task of interest, e.g., classification
    eps_syn = compute_model_diff(net, net_copy) # evalute how far the current model is away from the initial point

    net_eval = copy.deepcopy(net) # avoid unintended changes
    opt_eval = get_global_opt(args, net_eval)
    net_eval.eval()

    grad_real, grad_syn = None, None

    # compute statistics for real training data
    loss_avg, correct = 0, 0
    for img, lbl in test_loader:
        img, lbl = img.to(args.device), lbl.to(args.device)
        output = net_eval(img)
        # compute loss
        loss = metric(output, lbl)
        loss_avg += loss.item()/len(test_loader)

        # compute gradients
        if record_grad:
            opt_eval.zero_grad()
            loss.backward()
            grad = accumulate_grads(net_eval)
            if grad_real is None:
                grad_real = grad / len(test_loader)
            else:
                grad_real += grad / len(test_loader)

        # compute aurracy
        pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
        correct += pred.eq(lbl.view_as(pred)).sum().item()
    real_loss = loss_avg
    real_acc = 100. * correct / len(test_loader.dataset)

    # compute statistics for synthetic training data
    loss_avg, correct = 0, 0
    for img, lbl in syn_loader:
        img, lbl = img.to(args.device), lbl.to(args.device)
        output = net_eval(img)
        # compute loss
        loss = metric(output, lbl)
        loss_avg += loss.item()/len(syn_loader)

        # compute gradients
        if record_grad:
            opt_eval.zero_grad()
            loss.backward()
            grad = accumulate_grads(net_eval)
            if grad_syn is None:
                grad_syn = grad / len(syn_loader)
            else:
                grad_syn += grad / len(syn_loader)

        # compute aurracy
        pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
        correct += pred.eq(lbl.view_as(pred)).sum().item()
    syn_loss = loss_avg
    syn_acc = 100. * correct / len(syn_loader.dataset)

    if real_loader is not None:
        # compute statistics for real testing data
        loss_avg, correct = 0, 0
        for img, lbl in real_loader:
            img, lbl = img.to(args.device), lbl.to(args.device)
            output = net_eval(img)
            # compute loss
            loss = metric(output, lbl)
            loss_avg += loss.item()/len(real_loader)

            # compute aurracy
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(lbl.view_as(pred)).sum().item()
        test_loss = loss_avg
        test_acc = 100. * correct / len(real_loader.dataset)

    # MSE as gradient errors
    if record_grad:
        grad_error = np.sum((grad_real - grad_syn) ** 2)
        grad_cos = distance.cosine(grad_real, grad_syn)
    else:
        grad_error, grad_cos = None, None

    if real_loader is None:
        return eps_syn, real_loss, syn_loss, real_acc, syn_acc, grad_error, grad_cos#, grad_real, grad_syn
    else:
        return eps_syn, real_loss, syn_loss, real_acc, syn_acc, test_loss, test_acc, grad_error, grad_cos