import torch
from src.supervised_gcal.utils.functions import afferent_normalize
from src.supervised_gcal.utils.math import normalize
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights


class ModuleDecorator(torch.nn.Module):
    def __init__(self, linear):
        super(ModuleDecorator, self).__init__()
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError("Can only apply circular mask to Linear, type {}".format(type(linear)))
        self.linear = linear

    def __getattr__(self, s):
        try:
            x = super(ModuleDecorator, self).__getattr__(s)
        except AttributeError:
            pass
        else:
            return x
        x = self.linear.__getattr__(s)
        return x

    def forward(self, *input):
        return self.linear(*input)


class CircularMaskDecorator(ModuleDecorator):
    def __init__(self, linear, r):
        super(CircularMaskDecorator, self).__init__(linear)
        self.radius = r
        apply_circular_mask_to_weights(matrix=self.weight.data, radius=r)


class NormalizeDecorator(ModuleDecorator):
    def __init__(self, linear):
        super(NormalizeDecorator, self).__init__(linear)
        normalize(matrix=self.weight.data)


class AfferentNorm(torch.nn.Module):
    def __init__(self, strength, radius):
        super(AfferentNorm, self).__init__()
        self.radius = radius
        self.strength = strength

    def forward(self, afferent_input, activation):
        return afferent_normalize(strength=self.strength,
                                  afferent_input=afferent_input,
                                  activation=activation,
                                  radius=self.radius)


class CircularMask(torch.nn.Module):
    def __init__(self, radius):
        super(CircularMask, self).__init__()
        self.radius = radius

    def forward(self, input):
        # TODO: change to not in_place
        apply_circular_mask_to_weights(matrix=input, radius=self.radius)