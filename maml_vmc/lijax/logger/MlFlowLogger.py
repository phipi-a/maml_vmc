from contextlib import contextmanager
import os

import mlflow
from mlflow.entities import Metric

from maml_vmc.lijax.logger.Logger import Logger


class MlFlowLogger(Logger):
    def __init__(
        self,
        experiment_name: str = None,
        run_name: str = None,
        tracking_uri: str = None,
        logging_dir: str = "./logs",
        use_logger: bool = False,
        log_every: int = 1,
    ):
        super().__init__(use_logger, log_every, logging_dir)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.current_step = 0

    @contextmanager
    def _start_run(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            self.mlflow_client = mlflow.tracking.MlflowClient()
            run_id = mlflow.active_run().info.run_id
            self.run_id = run_id
            self.run_logging_dir = os.path.join(
                self.logging_dir, self.experiment_name, run_id
            )
            os.makedirs(self.run_logging_dir, exist_ok=True)
            yield

    def _to_metric(self, key: str, value: float):
        return Metric(key=key, value=value, timestamp=0, step=self.current_step)

    def _log_batch(self, metrics: list):
        self.mlflow_client.log_batch(run_id=self.run_id, metrics=metrics)

    def _get_logging_dir(self):
        return self.run_logging_dir

    def _log_config(self, params, config_file_path):
        mlflow.log_params(params)
        mlflow.log_artifact(config_file_path)
