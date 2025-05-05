import pickle
from jsonargparse import Namespace


class SelfStateClass:
    def __hash__(self):
        return hash(self.self_state)

    def __eq__(self, other):
        return self.self_state == other.self_state


def dict_to_namespace(d):
    if isinstance(d, dict):
        d = {k: dict_to_namespace(v) for k, v in d.items()}
        return Namespace(**d)
    elif isinstance(d, list):
        d = [dict_to_namespace(v) for v in d]
        return d
    else:
        return d


def load_training_state(path):
    state = pickle.load(open(path, "rb"))
    return state
