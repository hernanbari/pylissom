import copy

from sklearn.model_selection import GroupKFold

# from src.nn.utils import images as images

from src.utils.datasets import subj_indep_train_test_samplers
from src.utils.helpers import save_model
from src.utils.pipeline import Pipeline
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class CVSubjectIndependent(object):
    def __init__(self, ck_dataset, k=5):
        other_idxs, self.test_idxs = subj_indep_train_test_samplers(ck_dataset.subjs, 1 - 1 / k)
        self.folds = self._generate_folds(k - 1, ck_dataset, other_idxs)

    def train_val_samplers(self):
        return self.folds

    def test_sampler(self):
        return SubsetRandomSampler(self.test_idxs)

    @staticmethod
    def _generate_folds(k, ck_dataset, other_idxs):
        kf = GroupKFold(n_splits=k)
        return [(SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)) for train_index, val_index in
                kf.split(ck_dataset.X[other_idxs], ck_dataset.y[other_idxs],
                         ck_dataset.subjs[other_idxs])]


def run_cross_validation(model_fn, ck_dataset, cv_sampler, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    best_accuracy = 0
    best_model = None
    for fold, (train_sampler, val_sampler) in enumerate(cv_sampler.train_val_samplers()):
        train_loader = DataLoader(ck_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(ck_dataset, batch_size=args.batch_size, sampler=val_sampler, **kwargs)
        model, optimizer, loss_fn = model_fn()
        pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval, dataset_len=args.dataset_len,
                            cuda=args.cuda, prefix='fold_' + str(fold))
        if args.save_images:
            model[0].register_forward_hook(lambda *x: images.generate_images(*x, prefix='fold_' + str(fold)))
        # TODO: Change epochs to 0
        for epoch in range(1, args.epochs + 1):
            pipeline.train(train_data_loader=train_loader, epoch=epoch)
            curr_accuracy = pipeline.test(test_data_loader=val_loader, epoch=epoch)

        if best_model is None or curr_accuracy > best_accuracy:
            best_model = copy.deepcopy(model.state_dict())
            best_accuracy = curr_accuracy
    fold = 'test'
    model, optimizer, loss_fn = model_fn()
    model.load_state_dict(best_model)
    save_model(model)
    if args.save_images:
        model[0].register_forward_hook(lambda *x: images.generate_images(*x, prefix='fold_' + str(fold)))
    test_sampler = cv_sampler.test_sampler()
    test_loader = DataLoader(ck_dataset, batch_size=args.batch_size, sampler=test_sampler, **kwargs)
    pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval, dataset_len=args.dataset_len,
                        cuda=args.cuda, prefix='fold_' + str(fold))
    pipeline.test(test_data_loader=test_loader, epoch=1)
