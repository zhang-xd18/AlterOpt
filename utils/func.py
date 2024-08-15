from copy import deepcopy

__all__ = ['set_model_']


def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return
