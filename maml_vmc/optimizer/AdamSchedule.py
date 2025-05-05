import optax
from maml_vmc import lijax


class AdamSchedule(lijax.Optimizer):
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.scheduler = optax.schedules.exponential_decay(
            init_value=lr, transition_steps=500, decay_rate=0.5, transition_begin=0
        )

    def get_optimizer(self):
        return optax.adamw(self.scheduler, eps_root=1e-8)
