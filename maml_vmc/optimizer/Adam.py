import optax
from maml_vmc import lijax


class Adam(lijax.Optimizer):
    def __init__(self, lr=1e-3):
        self.lr = lr

    def get_optimizer(self):
        return optax.adamw(self.lr)
