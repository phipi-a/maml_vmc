import optax
from maml_vmc import lijax
import optax


def inverse_time_decay(lr_init, lr_decay_time, num_batches):
    def schedule(count):
        return lr_init / (1 + (count / num_batches) / lr_decay_time)

    return schedule


# Erstellen des Lernraten-Schedulers

# Erstellen des Optimierers mit dem benutzerdefinierten Lernraten-Scheduler


class Erwin(lijax.Optimizer):
    def __init__(self, lr: float, lr_decay_time: float, num_batches: int = 4):
        # Beispielwerte für die initiale Lernrate und die Abfallzeit
        self.lr_init = lr
        self.lr_decay_time = lr_decay_time
        self.scheduler = inverse_time_decay(lr, lr_decay_time, num_batches)

    def get_optimizer(self):
        optimizer = optax.adam(self.scheduler)
        return optimizer
