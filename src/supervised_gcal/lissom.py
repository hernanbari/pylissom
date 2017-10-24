from src.supervised_gcal.layer import AbstractLayer
from src.supervised_gcal.lgn_layer import LGNLayer
from src.supervised_gcal.reduced_lissom import ReducedLissom


class Lissom(AbstractLayer):
    def __init__(self,  input_shape, self_shape, lgn_shape, lgn_params=None, v1_params=None):
        self.lgn_shape = lgn_shape
        if lgn_params is None:
            self.lgn_params = {}
        if v1_params is None:
            self.v1_params = {}
        self.lgn_activation = None
        super(Lissom, self).__init__(input_shape, self_shape)

    def _setup_weights(self):
        self.on = LGNLayer(input_shape=self.input_shape, self_shape=self.lgn_shape, on=True, **self.lgn_params)
        self.off = LGNLayer(input_shape=self.input_shape, self_shape=self.lgn_shape, on=False, **self.lgn_params)
        self.v1 = ReducedLissom(input_shape=self.lgn_shape, self_shape=self.self_shape, **self.v1_params)

    def forward(self, retina):
        self.input = retina
        on_output = self.on(retina)
        off_output = self.off(retina)
        self.lgn_activation = on_output + off_output
        self.activation = self.v1(self.lgn_activation)
        return self.activation

    def register_forward_hook(self, hook):
        handles = []
        for m in self.children():
            handles.append(m.register_forward_hook(hook))
        return handles
