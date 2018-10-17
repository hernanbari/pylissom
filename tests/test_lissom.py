from tests.conftest import copy_module
from pylissom.nn.modules import Cortex, DifferenceOfGaussiansLinear, LGN, ReducedLissom, Lissom


class TestLissom(object):
    """
    Test class for the lissom module
    """

    def test_reduced_lissom_steps_1(self, rlissom, gaussian_variable):
        if gaussian_variable.size()[1] != rlissom.in_features:
            return
        if rlissom.settling_steps == 1:
            rlissom_copy = copy_module(rlissom)
            step_0_activation, afferent_activation = self.step_zero_activation(gaussian_variable, rlissom_copy)
            step_1_activation = self.step_n_activation(step_0_activation, afferent_activation, rlissom_copy)
            assert rlissom(gaussian_variable) == step_1_activation
        elif rlissom.settling_steps == 0:
            rlissom_copy = copy_module(rlissom)
            step_zero_activation, _ = self.step_zero_activation(gaussian_variable, rlissom_copy)
            assert rlissom(gaussian_variable) == step_zero_activation
        else:
            rlissom_copy = copy_module(rlissom)
            step_n_activation, afferent_activation = self.step_zero_activation(gaussian_variable, rlissom_copy)
            for _ in range(rlissom.settling_steps):
                step_n_activation = self.step_n_activation(step_n_activation, afferent_activation, rlissom_copy)
            assert rlissom(gaussian_variable) == step_n_activation

    def step_zero_activation(self, gaussian_variable, rlissom):
        afferent_module = rlissom.afferent_module
        piecewise_sigmoid = rlissom.piecewise_sigmoid
        # This is t=0
        afferent_activation = rlissom.afferent_strength * afferent_module(gaussian_variable)
        activation = piecewise_sigmoid(afferent_activation)
        return activation, afferent_activation

    def step_n_activation(self, previous_act, aff_act, rlissom):
        sum_activations = aff_act \
                          + rlissom.excitatory_strength * rlissom.excitatory_module(previous_act) \
                          - rlissom.inhibitory_strength * rlissom.inhibitory_module(previous_act)
        return rlissom.piecewise_sigmoid(sum_activations)

    def test_cortex(self):
        pass

    def test_difference_of_gaussians_linear(self):
        pass

    def test_lgn(self):
        pass
