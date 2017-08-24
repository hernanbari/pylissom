import tensorflow as tf
from tensorflow.python.training import optimizer


class HebbianOptimizer(optimizer.Optimizer):

    def __init__(self, use_locking=False, name='Hebbian'):
        super(HebbianOptimizer).__init__(use_locking, name)

    def _create_slots(self, var_list):
        pass

    def _apply_dense(self, grad, var):
        # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)
        def _debug_func(opt, grad, var):
            import ipdb; ipdb.set_trace()
            return False
        debug_op = tf.py_func(_debug_func, [self, grad, var], [tf.bool])
        return debug_op

