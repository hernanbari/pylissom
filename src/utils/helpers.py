import torch


def save_model(model, optimizer=None, fname='best_model.pth.tar'):
    state = {'model': model}
    if optimizer is not None:
        state.update({
            'optimizer': optimizer})
    torch.save(state, fname)


def load_model(fname='model.pth.tar'):
    state = torch.load(fname)
    return state['model'], state['optimizer'] if 'optimizer' in state else None
