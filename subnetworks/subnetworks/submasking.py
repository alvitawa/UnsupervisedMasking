import copy
import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import autograd, nn
from torch.nn import ParameterList, ParameterDict

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, mode):
        out = scores.clone()

        if mode == 'topk':
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())

            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

        elif mode == 'threshold':
            # Can't do out = scores > 0 because the gradient is lost then
            out[out <= k] = 0
            out[out > k] = 1

        elif mode == 'flip':
            out[out <= k] = -1
            out[out > k] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None

class MaskedConstantLayer(nn.Module):
    def __init__(self, in_ft, out_ft, score_init, positive=True, negative=False):
        super().__init__()
        if positive:
            self.WP = nn.Parameter(torch.ones(in_ft, out_ft) / np.sqrt(in_ft))
            self.SP = nn.Parameter(torch.ones(in_ft, out_ft)*score_init)
        if negative:
            self.WN = nn.Parameter(torch.ones(in_ft, out_ft) / np.sqrt(in_ft))
            self.SN = nn.Parameter(torch.ones(in_ft, out_ft)*score_init)
        assert positive or negative
        self.positive = positive
        self.negative = negative

    def forward(self, x):
        if self.positive:
            MP = GetSubnet.apply(self.SP, 0, 'threshold')
            xP = torch.mm(x, self.WP * MP) / torch.sqrt(MP.mean())
        if self.negative:
            MN = GetSubnet.apply(self.SN, 0, 'threshold')
            xN = torch.mm(x, self.WN * MN) / torch.sqrt(MN.mean())
        if self.positive and self.negative:
             out = (xP + xN) / 2
        elif self.positive:
             out = xP
        elif self.negative:
             out = xN
        return out

class MaskedLinear(nn.Module):
    def __init__(self, in_ft, out_ft, score_init):
        super().__init__()
        self.W = nn.Linear(in_ft, out_ft, bias=False)
        self.W.weight.requires_grad = False
        self.S = nn.Parameter(torch.ones(out_ft, in_ft)*score_init)

    def forward(self, x):
        M = GetSubnet.apply(self.S, 0, 'threshold')
        out = x @ (self.W.weight * M).T / torch.sqrt(M.mean())
        # import pdb; pdb.set_trace()
        return out


def default_scores_init(scores, _weights):
    # Initialize scores
    if len(scores.shape) < 2:
        # Bias
        scores.data.fill_(math.sqrt(5))
    else:
        nn.init.kaiming_uniform_(scores, a=math.sqrt(5))

def positive_scores_init(scores):
    scores.data.fill_(1.0)

def normal_scores_init(mean=0.0, std=0.01):
    return lambda scores, weights: scores.data.normal_(mean, std)

def magnitude_scores_init(mean=0.0, std=0.01, shuffle=False):
    def f(scores, weights):
        if shuffle:
            # Shuffle all weights with each other in the weight matrix of arbitrary shape
            weights = weights.flatten()[torch.randperm(weights.numel())].view(weights.shape)
        scores.data.copy_(torch.abs(weights) + torch.normal(mean, std, size=scores.shape, device=scores.device))

    return f
    # return lambda scores, weights: scores.data.copy_(torch.abs(weights) + torch.normal(mean, std, size=weights.shape, device=weights.device))


def set_attribute(obj, attr_name, value):
    variable = obj
    attr_path = attr_name.split('.')
    for variable_name in attr_path[:-1]:
        variable = getattr(variable, variable_name)
    setattr(variable, attr_path[-1], value)



class SubmaskedModel(torch.nn.Module):
    def __init__(self, model, parameter_selection=lambda name, param: 'mask' if param.requires_grad else 'freeze', scale=True, scores_init=default_scores_init, test_input=None, prune_criterion='threshold', k=lambda epoch: 0, shell_mode='copy'):
        """
            :param shell_mode: 'copy' or 'replace'. If 'copy', the masked weights will be copied into the model shell on each
            forward pass. If 'replace', the tensors themselves will be replaced through setattr. 'copy' mode does not support
            multiple forward passes without a backward pass in between. 'replace' mode may break if parameters are stored
            unconventionally.
        """
        super().__init__()
        # object.__setattr__(self, 'model_shell', model)
        self.model_shell = model

        self.scale = scale

        self.test_input = test_input

        self.prune_criterion = prune_criterion
        self.k = k

        self.shell_mode = shell_mode

        self.masked_params = []
        self.scores = ParameterDict()
        self.data = ParameterDict()
        for name, shell_param in self.model_shell.named_parameters():

            selection = parameter_selection(name, shell_param)

            if type(selection) is str:
                # 3 options: Freeze, train weights, or train masks
                mode = selection
                slot = 'default'
            else:
                mode, slot = selection

            if mode == 'freeze' or mode == 'mask':
                # Turn off all gradients for the model parameters
                shell_param.requires_grad = False
            elif mode == 'train':
                # Train the weights
                shell_param.requires_grad = True
            else:
                raise ValueError('Unknown parameter selection mode: {}'.format(mode))

            if mode == 'mask':
                # Create a score for each parameter
                scores_ = torch.nn.Parameter(torch.zeros_like(shell_param))

                self.masked_params.append((name, shell_param.data.clone().detach(), shell_param, scores_, slot))

                # Put it on a parameter list, so it is registered as a parameter
                name_ = name.replace('.', '_')
                self.scores[name_] = scores_
                self.data[name_] = shell_param.data.clone().detach()

        for name_ in self.data.keys():
            data, scores_ = self.data[name_], self.scores[name_]
            data.requires_grad = False
            scores_init(scores_, data)


    def get_masks(self, ctx=None):
        if ctx is None:
            ctx = {}
        return list(GetSubnet.apply(s, self.k(ctx), self.prune_criterion) for s in self.scores.values())

    def forward(self, *args, ctx=None, **kwargs):
        if ctx is None:
            ctx = {}

        for name, data, param_shell, scores, slot in self.masked_params:
            # Get the current mask
            mask = GetSubnet.apply(scores, self.k(ctx), self.prune_criterion)
            if self.scale:
                # Scale the weights to keep the output variance constant
                data_scaled = data * torch.sqrt(1/mask.mean())
            else:
                data_scaled = data
            if self.shell_mode == 'copy':
                # Detach the gradients from the previous forward pass
                param_shell.detach_()
                # Copy the data over into the shell parameter
                param_shell.copy_(data_scaled * mask)
            elif self.shell_mode == 'replace':
                # This is a bit tricky, the parameter needs to be replaced in a such a way that the gradients are not
                # lost. Since the Parameter constructor does not have a grad_fn, we do not directly initialize it
                # with the masked weights, instead we dummy-initialize it and then copy the data over with copy_, which
                # does have a grad_fn.
                masked_weights = data_scaled * mask
                parameter = torch.nn.Parameter(torch.empty_like(masked_weights), requires_grad = False)
                set_attribute(self.model_shell, name, parameter)
                parameter.copy_(masked_weights)
                # Dont need to detach the gradients in 'replace' mode, since the old parameter is not used

        # Now just forward through the model copy
        result = self.model_shell(*args, **kwargs)
        return result

    def analyze(self, ctx=None, depth_analysis=False):
        if ctx is None:
            ctx = {}

        analysis = {}
        if len(self.masked_params) == 0:
            return analysis

        k = self.k(ctx)
        analysis['prune_k'] = k

        # Get the mean mask across all parameters
        masks = self.get_masks(ctx)
        mean_mask = torch.cat([m.flatten() for m in masks]).mean()
        std_mask = torch.cat([m.flatten() for m in masks]).std()
        total_on = torch.cat([m.flatten() for m in masks]).sum()
        total_off = torch.cat([m.flatten() for m in masks]).numel() - total_on
        analysis['mean_mask'] = mean_mask
        analysis['std_mask'] = std_mask
        analysis['total_on'] = total_on
        analysis['total_off'] = total_off


        # Get the mean score across all parameters
        scores = list(self.scores.values())
        mean_score = torch.cat([s.flatten() for s in scores]).mean()
        std_score = torch.cat([s.flatten() for s in scores]).std()
        max_score = torch.cat([s.flatten() for s in scores]).max()
        min_score = torch.cat([s.flatten() for s in scores]).min()
        analysis['mean_score'] = mean_score
        analysis['std_score'] = std_score
        analysis['max_score'] = max_score
        analysis['min_score'] = min_score

        # if ctx.get('epoch', None) == 3:
        #     import pdb; pdb.set_trace()

        # Same but per slot
        slots = set([slot for _, _, _, _, slot in self.masked_params])
        for slot in slots:
            slot_mean_mask = torch.cat([m.flatten() for m, (_, _, _, _, sl) in zip(masks, self.masked_params) if sl == slot]).mean()
            slot_std_mask = torch.cat([m.flatten() for m, (_, _, _, _, sl) in zip(masks, self.masked_params) if sl == slot]).std()
            slot_total_on = torch.cat([m.flatten() for m, (_, _, _, _, sl) in zip(masks, self.masked_params) if sl == slot]).sum()
            slot_total_off = torch.cat([m.flatten() for m, (_, _, _, _, sl) in zip(masks, self.masked_params) if sl == slot]).numel() - slot_total_on
            analysis['mean_mask_{}'.format(slot)] = slot_mean_mask
            analysis['std_mask_{}'.format(slot)] = slot_std_mask
            analysis['total_on_{}'.format(slot)] = slot_total_on
            analysis['total_off_{}'.format(slot)] = slot_total_off

            slot_mean_score = torch.cat([s.flatten() for _, _, _, s, sl in self.masked_params if sl == slot]).mean()
            slot_std_score = torch.cat([s.flatten() for _, _, _, s, sl in self.masked_params if sl == slot]).std()
            slot_max_score = torch.cat([s.flatten() for _, _, _, s, sl in self.masked_params if sl == slot]).max()
            slot_min_score = torch.cat([s.flatten() for _, _, _, s, sl in self.masked_params if sl == slot]).min()
            analysis['mean_score_{}'.format(slot)] = slot_mean_score
            analysis['std_score_{}'.format(slot)] = slot_std_score
            analysis['max_score_{}'.format(slot)] = slot_max_score
            analysis['min_score_{}'.format(slot)] = slot_min_score

        return analysis


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img