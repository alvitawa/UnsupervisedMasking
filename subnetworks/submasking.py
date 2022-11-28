import copy
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import autograd, nn
from torch.nn import ParameterList, ParameterDict

from volt import util


def traverse_leaves(root, fn, seen=None, child=None, depth=0):
    if seen is None:
        seen = set()
    seen.add(root)
    parents = root.next_functions
    if len(parents) == 0:
        return fn(root, child, depth)
    for parent, i in parents:
        if parent in seen:
            continue
        if parent is not None:
            traverse_leaves(parent, fn, seen, root, depth - 1)


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

def default_scores_init(scores):
    # Initialize scores
    if len(scores.shape) < 2:
        # Bias
        scores.data.fill_(math.sqrt(5))
    else:
        nn.init.kaiming_uniform_(scores, a=math.sqrt(5))

def positive_scores_init(scores):
    scores.data.fill_(1.0)

def normal_scores_init(mean=0.0, std=0.01):
    return lambda scores: scores.data.normal_(mean, std)



class SubmaskedModel(torch.nn.Module):
    def __init__(self, model, parameter_selection=lambda name, param: 'mask' if param.requires_grad else 'freeze', scale=True, scores_init=default_scores_init, test_input=None, prune_criterion='threshold', k=lambda epoch: 0):
        super().__init__()
        self.model_shell = model

        self.scale = scale

        self.test_input = test_input

        self.prune_criterion = prune_criterion
        self.k = k

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

        for data in self.data.values():
            data.requires_grad = False

        for scores_ in self.scores.values():
            scores_init(scores_)

    def analyze(self, ctx=None):
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

        if self.test_input is None:
            return

        prev = torch.is_grad_enabled()
        mode = self.training

        # Temporarily re-enable gradients to get the computation graph
        torch.set_grad_enabled(True)
        self.train()
        result = self.forward(self.test_input, ctx=ctx)


        # The metrics dont work during validation sometimes
        if result.grad_fn is not None:

            param_map = {}
            for name, data, param_shell, scores, slot in self.masked_params:
                param_map[id(scores)] = slot

            slot_ratios = {}

            def f(root, child, depth):
                slot = param_map.get(id(root.variable), None)
                if slot is not None:
                    mask = GetSubnet.apply(root.variable, self.k(ctx), self.prune_criterion)
                    ratio = mask.mean().item()
                    slot_ratios.setdefault(slot, dict()).setdefault(depth, []).append(ratio)

            traverse_leaves(result.grad_fn, f)

            analysis['data'] = str(slot_ratios)

            for slot, ratios in slot_ratios.items():
                ordered_ratios = sorted(ratios.items(), key=lambda x: x[0])
                y0 = [min(r[1]) for r in ordered_ratios]
                y1 = [max(r[1]) for r in ordered_ratios]
                x = list(range(len(y0)))

                fig = plt.figure()
                plt.title(slot)
                plt.fill_between(x, y0, y1, alpha=0.6)
                plt.plot(x, (np.array(y0) + np.array(y1)) / 2)
                plt.scatter(x, (np.array(y0) + np.array(y1)) / 2)
                plt.xlabel('Depth')
                plt.ylabel('Ratio')
                plot = util.fig2img(fig)
                plt.close(fig)

                analysis['plot_{}'.format(slot)] = plot


        # Restore the previous state of grad
        torch.set_grad_enabled(prev)
        self.train(mode)
        return analysis

    def get_masks(self, ctx=None):
        if ctx is None:
            ctx = {}
        return list(GetSubnet.apply(s, self.k(ctx), self.prune_criterion) for s in self.scores.values())

    def forward(self, *args, ctx=None, **kwargs):
        # Replace the parameters of the model copy with the masked versions
        # for param, param_shell, mask in zip(self.model.parameters(), self.model_shell.parameters(), masks):
        if ctx is None:
            ctx = {}
        for name, data, param_shell, scores, slot in self.masked_params:
            # Detach the gradients from the previous forward pass
            mask = GetSubnet.apply(scores, self.k(ctx), self.prune_criterion)
            param_shell.detach_()
            if self.scale:
                data_scaled = data * torch.sqrt(1/mask.mean())
            else:
                data_scaled = data
            param_shell.copy_(data_scaled * mask)
        # Now just forward through the model copy
        result = self.model_shell(*args, **kwargs)
        return result
