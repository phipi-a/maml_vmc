from contextlib import contextmanager
import os

import wandb

from maml_vmc.lijax.logger.Logger import Logger


class WandbLogger(Logger):
    def __init__(
        self,
        project: str = "maml_vmc",
        run_name: str = None,
        logging_dir: str = "./logs",
        use_logger: bool = False,
        log_every: int = 1,
    ):
        super().__init__(use_logger, log_every, logging_dir)
        self.project = project
        self.run_name = run_name
        self.current_step = 0

    @contextmanager
    def _start_run(self):

        self.run = wandb.init(
            dir=self.logging_dir, project=self.project, name=self.run_name
        )
        # wandb.run.log_code(
        #     ".",
        #     exclude_fn=lambda x: "/logs/" in x,
        # )
        run_logging_dir = os.path.join(self.logging_dir, "runs", self.run.id)
        os.makedirs(run_logging_dir, exist_ok=True)
        self.run_logging_dir = run_logging_dir
        yield
        self.run.finish()

    def _to_metric(self, key: str, value: float):
        return {key: value}, self.current_step

    def _log_batch(self, metrics: list):
        for metric, step in metrics[:-1]:
            wandb.log(metric, commit=False, step=step)
        last_metric, last_step = metrics[-1]
        wandb.log(last_metric, commit=True, step=last_step)

    def _get_logging_dir(self):
        return self.run_logging_dir

    def _log_config(self, params, config_file_path):
        wandb.config.update(params)
        base_path = "/".join(config_file_path.split("/")[:-2])
        wandb.save(config_file_path, base_path=base_path)
