from contextlib import contextmanager
import os
import pickle
from typing import Dict, Tuple

from jsonargparse import dict_to_namespace
import loguru
from tqdm import tqdm
import yaml


class Logger:
    def __init__(
        self,
        use_logger: bool = False,
        log_every: int = 1,
        logging_dir: str = "./logs",
    ):
        self.use_logger = use_logger
        self.bar_metrics = {}
        self.metrics_batch = []
        self.log_every = log_every
        self.logging_dir = logging_dir

        self.current_step = 0

    @contextmanager
    def _start_run(self):
        raise NotImplementedError

    def _to_metric(self, key: str, value: float):
        raise NotImplementedError

    def _log_batch(self, metrics: list):
        raise NotImplementedError

    def _get_logging_dir(self):
        raise NotImplementedError

    def _log_config(self, params, config_file_path):
        raise NotImplementedError

    def init_bar(self, num_steps):
        self.bar = tqdm(range(num_steps), dynamic_ncols=True)

    def step_finished(self):
        self.current_step += 1

        self.bar.set_postfix(self.bar_metrics)
        if self.current_step % self.log_every == 0 and self.use_logger:
            self._log_batch(metrics=self.metrics_batch)
            self.metrics_batch = []
        self.bar.update(1)
        if self.current_step == self.bar.total:
            self.bar.close()

    def get_logging_dir(self):
        if self.use_logger:
            return self._get_logging_dir()
        else:
            return self.logging_dir

    def log_params(self, save_config_fn: callable):
        if self.use_logger:
            config_file_path = os.path.join(self.get_logging_dir(), "config.yaml")

            save_config_fn(path=config_file_path)
            with open(config_file_path, "r") as f:
                config = yaml.safe_load(f)

            config = config["fit"]

            with open(config_file_path, "w") as f:
                yaml.dump(config, f)

            ns_config = dict_to_namespace(config)
            flat_config = ns_config.as_flat()
            flat_config_dict = vars(flat_config)

            self._log_config(params=flat_config_dict, config_file_path=config_file_path)

    def log_metrics(
        self,
        metrics: Dict[str, Tuple[float, bool]],
    ):
        for key, (value, is_bar_metric) in metrics.items():
            self.metrics_batch.append(
                self._to_metric(
                    key=key,
                    value=value,
                )
            )
            if is_bar_metric:
                self.bar_metrics[key] = value

    def store_training_state(self, state):
        state_file_dir = os.path.join(self.get_logging_dir(), "checkpoints")
        if not os.path.exists(state_file_dir):
            os.makedirs(state_file_dir)
        state_file_path = os.path.join(state_file_dir, f"state_{self.current_step}.pkl")
        pickle.dump(state, open(state_file_path, "wb"))
        loguru.logger.info(f"Stored training state at {state_file_path}")

    def set_current_step(self, current_step):
        self.current_step = current_step
        self.bar.n = current_step

    @contextmanager
    def start_run(self):
        if self.use_logger:
            with self._start_run():
                yield
        else:
            yield
