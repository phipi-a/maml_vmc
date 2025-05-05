import optax


class Optimizer:
    def get_optimizer(self) -> optax.GradientTransformation:
        raise NotImplementedError
