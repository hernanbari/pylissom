import numpy as np
from sklearn.model_selection import ParameterGrid

from src.datasets.datasets import get_dataset
from src.models.models import get_lissom, get_supervised, get_reduced_lissom
from src.nn.utils import images as images
from src.utils.training.pipeline import Pipeline


class GridSearch(object):
    def __init__(self, model_fn, param_grid, train_loader=None, test_loader=None,
                 epochs=1, **kwargs):
        assert train_loader is not None or test_loader is not None
        self.model_fn = model_fn
        self.param_grid = ParameterGrid(param_grid)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.kwargs = kwargs

    def run(self):
        for counter, params in self.param_grid:
            model, optimizer, loss_fn = self.model_fn(params, counter)
            pipeline = Pipeline(model, optimizer, loss_fn)
            # TODO: Change epochs to 0
            for epoch in range(self.epochs + 1):
                if self.train_loader is not None:
                    pipeline.train(self.train_loader, epoch)
                if self.test_loader is not None:
                    pipeline.train(self.test_loader, epoch)
            print("Iteration", counter)
            print(params)


def run_lgn_grid_search(input_shape, lgn_shape, args):
    def model_fn(lgn_params, counter):
        lissom, optimizer, _ = get_lissom(input_shape, lgn_shape, (1, 1), lgn_params=lgn_params)

        def hardcoded_counter(self, input, output):
            self.batch_idx = counter
            images.generate_images(self, input, output)

        lissom.register_forward_hook(hardcoded_counter)
        return lissom, None, _

    param_grid = {'sigma_center': np.arange(0.1, 10, step=0.5),
                  'sigma_sorround': [1.5, 2, 3, 5, 8, 10],
                  'radius': [3, 4, 5, 8, 10, 15, 20]}
    test_loader = get_dataset(train=False, args=args)
    grid_search = GridSearch(model_fn, param_grid, test_loader=test_loader)
    grid_search.run()


def run_cortex_grid_search(input_shape, cortex_shape, args):
    def model_fn(v1_params, counter):
        lissom, optimizer, _ = get_reduced_lissom(input_shape, cortex_shape, v1_params=v1_params)

        # def hardcoded_counter(self, input, output):
        #     self.batch_idx = counter
        #     images.generate_images(self, input, output)
        #
        # lissom.register_forward_hook(hardcoded_counter)
        return lissom, optimizer, _

    param_grid = {'min_theta': [0, 0.2, 0.5], 'max_theta': [0.6, 0.8, 1.0], 'afferent_radius': [5, 10, 15, 20],
                  'excitatory_radius': [2, 5, 10, 15], 'inhibitory_radius': [5, 10, 15, 20, 25],
                  'settling_steps': 10, 'inhib_factor': [1, 1.2, 1.5, 2.0, 3], 'excit_factor': [1, 1.2, 1.5, 2.0, 3]}
    test_loader = get_dataset(train=False, args=args)
    grid_search = GridSearch(model_fn, param_grid, test_loader=test_loader)
    grid_search.run()


def run_lissom_grid_search(input_shape, lgn_shape, cortex_shape, args):
    def model_fn(v1_params, counter):
        v1_params['inhibitory_radius'] = v1_params['afferent_radius']
        lissom, optimizer, _ = get_lissom(input_shape, lgn_shape, cortex_shape, pruning_step=args.log_interval,
                                          final_epoch=args.epochs,
                                          v1_params=v1_params)

        def hardcoded_counter(self, input, output):
            self.batch_idx = counter
            images.generate_images(self, input, output)

        lissom.register_forward_hook(hardcoded_counter)
        return lissom, optimizer, _

    param_grid = {'afferent_radius': [3, 5, 10, 15, 20], 'excitatory_radius': [2, 4, 9, 14],
                  'inhib_factor': [1.0, 1.5, 3.0], 'excit_factor': [1.0, 1.5, 3.0]}
    train_loader = get_dataset(train=True, args=args)
    grid_search = GridSearch(model_fn, param_grid, train_loader=train_loader)
    grid_search.run()


# TODO: train lissom first and then net
def run_supervised_grid_search(input_shape, lgn_shape, cortex_shape, args):
    def model_fn(v1_params, counter):
        v1_params['inhibitory_radius'] = v1_params['afferent_radius']
        model, optimizer, loss_fn = get_supervised(input_shape, lgn_shape, cortex_shape, pruning_step=args.log_interval,
                                                   final_epoch=args.epochs)
        if args.save_images:
            def hardcoded_counter(self, input, output):
                images.logdir = args.logdir + '/counter_' + str(counter)
                images.generate_images(self, input, output)

            model[0].register_forward_hook(hardcoded_counter)

    param_grid = []
    params_names = ('afferent_radius', 'excitatory_radius', 'inhib_factor', 'excit_factor')
    for params_values in [([3], [2], [1], [1.5]), ([5], [2], [1], [1.5]), ([5], [4], [1], [1.5]), ([5], [4], [3], [3])]:
        param_grid.append(dict(zip(params_names, params_values)))
    test_loader = get_dataset(train=False, args=args)
    train_loader = get_dataset(train=True, args=args)
    grid_search = GridSearch(model_fn, param_grid, train_loader=train_loader, test_loader=test_loader)
    grid_search.run()

# LGN
# counter = 0
#
# test_loader = get_dataset(train=False, args=args)
# for sigma_center in np.arange(0.1, 10, step=0.5):
#     for sigma_sorround in [1.5, 2, 3, 5, 8, 10]:
#         for radius in [3, 4, 5, 8, 10, 15, 20]:
#             lgn_shape = (args.shape, args.shape)
#             model = FullLissom(input_shape, lgn_shape, (1, 1),
#                                lgn_params={'sigma_center': sigma_center, 'sigma_sorround': sigma_sorround,
#                                            'radius': radius})
#             pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval,
#                                 dataset_len=args.dataset_len,
#                                 cuda=args.cuda)
#             if args.save_images:
#                 def hardcoded_counter(self, input, output):
#                     self.batch_idx = counter
#                     images.generate_images(self, input, output)
#
#
#                 model.register_forward_hook(hardcoded_counter)
#             pipeline.test(test_data_loader=test_loader)
#             print("Iteration", counter)
#             print(sigma_center, sigma_sorround, radius)
#             counter += 1
# exit()

# Lissom
#
# counter = 0
# for afferent_radius in [3, 5, 10, 15, 20]:
#     for excitatory_radius in [2, 4, 9, 14]:
#         for inhib_factor in [1.0, 1.5, 3.0]:
#             for excit_factor in [1.0, 1.5, 3.0]:
#                 if excitatory_radius > afferent_radius:
#                     continue
#
#                 lgn_shape = (args.shape, args.shape)
#                 lissom_shape = (args.shape, args.shape)
#                 inhibitory_radius = afferent_radius
#                 params_list = [(name, eval(name)) for name in
#                                ['afferent_radius', 'excitatory_radius',
#                                 'inhib_factor', 'excit_factor']]
#                 model = FullLissom(input_shape, lgn_shape, lissom_shape,
#                                    v1_params=dict(params_list))
#                 optimizer = SequentialOptimizer(
#                     CortexHebbian(cortex_layer=model.v1),
#                     NeighborsDecay(cortex_layer=model.v1, pruning_step=args.log_interval, final_epoch=5)
#                 )
#                 pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval,
#                                     dataset_len=args.dataset_len,
#                                     cuda=args.cuda)
#                 if args.save_images:
#                     def hardcoded_counter(self, input, output):
#                         self.batch_idx = counter
#                         images.generate_images(self, input, output)
#
#
#                     model.register_forward_hook(hardcoded_counter)
#                 for epoch in range(1, args.epochs + 1):
#                     pipeline.train(train_data_loader=train_loader, epoch=epoch)
#                 print("Iteration", counter)
#                 print(params_list)
#                 counter += 1
# exit()


# Supervised
# counter = 0
# test_loader = get_dataset(train=False, args=args)
# train_loader = get_dataset(train=True, args=args)
# params_names = ('afferent_radius', 'excitatory_radius', 'inhib_factor', 'excit_factor')
# for params_values in [(3, 2, 1, 1.5), (5, 2, 1, 1.5), (5, 4, 1, 1.5), (5, 4, 3, 3)]:
#     params_dict = dict(zip(params_names, params_values))
#     lgn_shape = (args.shape, args.shape)
#     lissom_shape = (args.shape, args.shape)
#     inhibitory_radius = params_dict['afferent_radius']
#     lissom = FullLissom(input_shape, lgn_shape, lissom_shape,
#                         v1_params=params_dict)
#     net_input_shape = lissom.activation_shape[1]
#     net = torch.nn.Sequential(
#         torch.nn.Linear(net_input_shape, classes),
#         torch.nn.LogSoftmax()
#     )
#     loss_fn = torch.nn.functional.nll_loss
#     model = torch.nn.Sequential(
#         lissom,
#         net
#     )
#     optimizer = SequentialOptimizer(
#         CortexHebbian(cortex_layer=lissom.v1),
#         NeighborsDecay(cortex_layer=lissom.v1, pruning_step=args.log_interval, final_epoch=args.epochs),
#         torch.optim.SGD(net.parameters(), lr=0.1)
#     )
#     pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval,
#                         dataset_len=args.dataset_len,
#                         cuda=args.cuda)
#     if args.save_images:
#         def hardcoded_counter(self, input, output):
#             images.logdir = args.logdir + '/counter_' + str(counter)
#             images.generate_images(self, input, output)
#
#
#         lissom.register_forward_hook(hardcoded_counter)
#     for epoch in range(1, args.epochs + 1):
#         pipeline.train(train_data_loader=train_loader, epoch=epoch)
#         pipeline.test(test_data_loader=test_loader, epoch=epoch)
#     print("Iteration", counter)
#     print(params_dict)
#     counter += 1
# exit()
