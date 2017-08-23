import tensorflow as tf
from tensorflow.python.training import optimizer


class HebbianOptimizer(optimizer.Optimizer):
    def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
        raise NotImplementedError