import tensorflow as tf
from tensorflow.python.training import optimizer


class HebbianOptimizer(optimizer.Optimizer):
    # def get_slot(self, var, name):

    # def get_slot_names(self):

    # --------------
    # Utility methods for subclasses.
    # --------------

    # def _slot_dict(self, slot_name):
    #
    # def _get_or_make_slot(self, var, val, slot_name, op_name):

    # def _get_or_make_slot_with_initializer(self, var, initializer, shape, dtype,
    #                                        slot_name, op_name):
    #
    # def _zeros_slot(self, var, slot_name, op_name):

    def __init__(self, use_locking=False, name='Hebbian'):
        super().__init__(use_locking, name)

    def _prepare(self):
        import ipdb; ipdb.set_trace()
        pass

    def _create_slots(self, var_list):
        import ipdb; ipdb.set_trace()
        pass

    def _finish(self, update_ops, name_scope):
        import ipdb; ipdb.set_trace()
        super()._finish(update_ops, name_scope)


    def _apply_dense(self, grad, var):
        import ipdb; ipdb.set_trace()
        # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)
        def _debug_func(opt, grad, var):
            import ipdb; ipdb.set_trace()
            return False
        debug_op = tf.py_func(_debug_func, [grad, var], [tf.bool])
        return var

