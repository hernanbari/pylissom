from collections import OrderedDict

import torch
from torch.nn import Linear, Sequential, Module

from pylissom.math import normalize
from pylissom.nn.functional.functions import check_compatible_mul, check_compatible_add
from pylissom.nn.functional.weights import apply_circular_mask_to_weights, get_gaussian_weights
from pylissom.nn.modules.linear import GaussianCloudLinear, PiecewiseSigmoid


class Cortex(GaussianCloudLinear):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    where A is a Gaussian Cloud with a connective radius

    This module is primarily used to build a :py:class`ReducedLissom`

    Parameters:
        - **radius** -
        - **sigma** -
    """
    def __init__(self, in_features, out_features, radius, sigma=1.0):
        super(Cortex, self).__init__(in_features, out_features, sigma=sigma)
        self.radius = radius
        self.weight.data = normalize(
            apply_circular_mask_to_weights(self.weight.data,
                                           radius=radius))

    def __repr__(self):
        super_repr = super(Cortex, self).__repr__()[:-1]
        return super_repr \
               + ', ' + 'radius=' + str(self.radius) \
               + ')'


# # TODO: test
# class AfferentNormCortex(Sequential):
#     def __init__(self, in_features, out_features, radius, aff_norm_strength, sigma=1.0,
#                  cortex_cls=Cortex, aff_norm_cls=AfferentNorm):
#         self.sigma = sigma
#         self.aff_norm_strength = aff_norm_strength
#         self.radius = radius
#         self.out_features = out_features
#         self.in_features = in_features
#         layers = OrderedDict(
#             {'cortex': cortex_cls(in_features=in_features, out_features=out_features, radius=radius, sigma=sigma),
#              'aff_norm': aff_norm_cls(aff_norm_strength=aff_norm_strength, radius=radius)})
#         super(AfferentNormCortex, self).__init__(layers)


class DifferenceOfGaussiansLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`,
    where A is a Difference of Gaussians with a connective radius:

    .. math::

        \begin{equation*}
        \text{out}_ab = \sigma(\phi_L \sum_(xy) \text{input}_xy L_xy,ab)
        \end{equation*}

    Parameters:
        - **on** - Defines if the substraction goes sorround gaussian - center gaussian or the other way around
        - **radius** -
        - **sigma_surround** -
        - **sigma_center** -
    """

    # ASSUMES SQUARE MAPS
    # Order of diff(normalize(mask(gaussian)))) is important
    def __init__(self, in_features, out_features, on, radius, sigma_surround, sigma_center=1.0):
        super(DifferenceOfGaussiansLinear, self).__init__(in_features, out_features, bias=False)
        self.on = on
        self.sigma_surround = sigma_surround
        self.sigma_center = sigma_center
        self.radius = radius
        sigma_center_weights_matrix = normalize(
            apply_circular_mask_to_weights(get_gaussian_weights(in_features, out_features,
                                                                sigma_center), radius=radius))
        sigma_surround_weights_matrix = normalize(
            apply_circular_mask_to_weights(get_gaussian_weights(in_features, out_features,
                                                                sigma_surround), radius=radius))
        if on:
            diff = sigma_center_weights_matrix - sigma_surround_weights_matrix
        else:
            diff = sigma_surround_weights_matrix - sigma_center_weights_matrix
        self.weight = torch.nn.Parameter(diff)

    def __repr__(self):
        super_repr = super(DifferenceOfGaussiansLinear, self).__repr__()[:-1]
        return super_repr \
               + ', ' + 'sigma_surround=' + str(self.sigma_surround) \
               + ', ' + 'sigma_center=' + str(self.sigma_center) \
               + ', ' + 'radius=' + str(self.radius) \
               + ', ' + 'on=' + str(self.on) \
               + ')'


class Mul(Module):
    r"""Represents a layer than only multiplies the input by a constant, used in :py:class:`pylissom.nn.modules.LGN`"""
    def __init__(self, number):
        super(Mul, self).__init__()
        self.number = number

    def forward(self, input):
        return input * self.number

    def __repr__(self):
        return '*' + str(self.number)


class LGN(Sequential):
    r"""Represents an LGN channel, can be ON or OFF

    The transformation applied can be described as:

    .. math::

        \begin{equation*}
        \text{out}_ab = \sigma(\phi_L \sum_(xy) \text{input}_xy L_xy,ab)
        \end{equation*}

    where :math:`\sigma` is the piecewise sigmoid,
    :math:`N` is foo

    It inherits from :py:class:`Sequential` because an LGN is in essence a composition of several transformations

    * :attr:`afferent_module`

    Parameters:
        - **on** -
        - **radius** -
        - **sigma_surround** -
        - **sigma_center** -
        - **strength** -
        - **min_theta** -
        - **max_theta** -
    """
    def __init__(self, in_features, out_features, on, radius, sigma_surround,
                 sigma_center=1.0, min_theta=0.0, max_theta=1.0, strength=1.0,
                 diff_of_gauss_cls=DifferenceOfGaussiansLinear, pw_sigmoid_cls=PiecewiseSigmoid):
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.in_features = in_features
        self.out_features = out_features

        layers = OrderedDict({
            'diff_of_gaussians': diff_of_gauss_cls(in_features=in_features, out_features=out_features, on=on,
                                                   radius=radius, sigma_center=sigma_center,
                                                   sigma_surround=sigma_surround),
            'strength': Mul(strength),
            'piecewise_sigmoid': pw_sigmoid_cls(min_theta=min_theta, max_theta=max_theta)})
        super(LGN, self).__init__(layers)


class ReducedLissom(Module):
    r"""Represents a Reduced Lissom consisting of afferent, excitatory and inhibitory modules

    The transformation applied can be described as:

    .. math::

        \begin{equation*}
        n_ij = \sigma(s_ij + \phi_E \sum_(kl) n_kl (t-1) E_kl,ij - \phi_I \sum_(kl) n_kl (t-1) E_kl,ij I_lk,ij)
        \end{equation*}

    where :math:`\sigma` is the piecewise sigmoid,
    :math:`N` is foo

    * :attr:`afferent_module`

    Parameters:
        - **afferent_module** -
        - **excitatory_module** -
        - **inhibitory_module** -
        - **afferent_strength** -
        - **excitatory_strength** -
        - **inhibitory_strength** -
        - **min_theta** -
        - **max_theta** -
        - **settling_steps** -

    """

    def __init__(self, afferent_module, excitatory_module, inhibitory_module,
                 min_theta=1.0, max_theta=1.0, settling_steps=10,
                 afferent_strength=1.0, excitatory_strength=1.0, inhibitory_strength=1.0,
                 pw_sigmoid_cls=PiecewiseSigmoid):
        super(ReducedLissom, self).__init__()
        check_compatible_mul(afferent_module, inhibitory_module)
        check_compatible_mul(afferent_module, excitatory_module)
        self.in_features = afferent_module.in_features
        self.out_features = afferent_module.out_features
        self.inhibitory_module = inhibitory_module
        self.excitatory_module = excitatory_module
        self.afferent_module = afferent_module
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.settling_steps = settling_steps
        self.inhibitory_strength = inhibitory_strength
        self.excitatory_strength = excitatory_strength
        self.afferent_strength = afferent_strength
        self.piecewise_sigmoid = pw_sigmoid_cls(min_theta=min_theta, max_theta=max_theta)

    def forward(self, cortex_input):
        afferent_activation = self.afferent_strength * self.afferent_module(cortex_input)
        current_activation = self.piecewise_sigmoid(afferent_activation)
        for _ in range(self.settling_steps):
            excitatory_activation = self.excitatory_strength * self.excitatory_module(current_activation)
            inhibitory_activation = self.inhibitory_strength * self.inhibitory_module(current_activation)

            sum_activations = afferent_activation + excitatory_activation - inhibitory_activation

            current_activation = self.piecewise_sigmoid(sum_activations)
        return current_activation

    def __repr__(self):
        super_repr = super(ReducedLissom, self).__repr__()[:-1]
        return super_repr \
               + ', ' + str(self.in_features) + ' -> ' + str(self.out_features) \
               + ', ' + 'settling_steps=' + str(self.settling_steps) \
               + ', ' + 'afferent_strength=' + str(self.afferent_strength) \
               + ', ' + 'excitatory_strength=' + str(self.excitatory_strength) \
               + ', ' + 'inhibitory_strength=' + str(self.inhibitory_strength) \
               + ')'


class Lissom(Module):
    r"""Represents a Full Lissom, with ON/OFF channels and a V1 ( :py:class:`ReducedLissom` )

    The transformation applied can be described as:

    .. math::

        \begin{equation*}
        \text{out} = \text{v1}(\text{on}(input) + \text{off}(input))
        \end{equation*}

    Parameters:
        - **on** - an ON :py:class:`LGN` map
        - **off** - an OFF :py:class:`LGN` map
        - **v1** - a :py:class:`ReducedLissom` map

    Shape:
        - TODO
    """

    def __init__(self, on, off, v1):
        super(Lissom, self).__init__()
        check_compatible_add(on, off)
        check_compatible_mul(on, v1)
        self.v1 = v1
        self.off = off
        self.on = on
        self.in_features = on.in_features
        self.out_features = v1.out_features

    def forward(self, input):
        on_output = self.on(input)
        off_output = self.off(input)
        lgn_activation = on_output + off_output
        activation = self.v1(lgn_activation)
        return activation

    def __repr__(self):
        super_repr = super(Lissom, self).__repr__()[:-1]
        return super_repr \
               + ', ' + str(self.in_features) + ' -> ' + str(self.out_features) \
               + ')'


