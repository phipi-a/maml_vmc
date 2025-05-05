from maml_vmc.lijax.logger.Logger import Logger


class Trainer:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def fit(self):
        raise NotImplementedError

    def start(self, logger: Logger):
        self.logger = logger
        self.logger.init_bar(self.num_steps)
        self.fit()

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
