"""
Extends :py:mod:`torch.nn` with Lissom layers, split in the simpler Linear module and the higher-level Lissom module
"""
from pylissom.nn.modules.linear import *
from pylissom.nn.modules.lissom import *


def register_recursive_forward_hook(module, hook):
    r"""Adds a forward hook to all modules in module"""
    return [m.register_forward_hook(hook) for m in module.modules()]


def named_apply(mod, fn, prefix):
    r"""Like :py:func:`torch.nn.Module.apply` but with named children"""
    for k, m in mod.named_children():
        named_apply(m, fn, prefix + '.' + k)
    fn(mod, prefix)


def input_output_hook(module, input, output):
    module.input = input[0].clone()
    module.output = output.clone()


def register_recursive_input_output_hook(module):
    r"""Adds a hook to module so it saves in memory input and output in each forward pass"""
    return register_recursive_forward_hook(module, input_output_hook)
