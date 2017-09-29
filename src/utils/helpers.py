import torch


def save_model(model, optimizer=None, fname='best_model.pth.tar'):
    state = {'model_class': type(model),
             'state_dict': model.state_dict()}
    if optimizer is not None:
        state.update({
            'optimizer_class': type(optimizer),
            'optimizer': optimizer.state_dict()})
    torch.save(state, fname)


def load_model(fname='model.pth.tar'):
    state = torch.load(fname)
    model = state['model_class'].load_state_dict(state['state_dict'])
    optimizer = None
    if 'optimizer' in state:
        optimizer = state['optimizer_class'].load_state_dict(state['optimizer'])
    return model, optimizer