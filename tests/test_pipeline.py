from torch.utils.data import DataLoader

from pylissom.datasets import ThreeDotFaces
from pylissom.utils.training import Pipeline
from tests.conftest import get_dummy_lissom, get_dummy_lissom_hebbian


class TestPipeline(object):
    """
    Test class for the pipeline module
    """

    def test_pipeline(self):
        model = get_dummy_lissom()
        optimizer = get_dummy_lissom_hebbian(model)
        gaussians_inputs = ThreeDotFaces(size=int(model.in_features ** 0.5), length=10)
        train_loader = DataLoader(gaussians_inputs, shuffle=True, pin_memory=False)
        pipe = Pipeline(model, optimizer)
        pipe.train(train_loader, epoch=0)
        assert 1 == 1
