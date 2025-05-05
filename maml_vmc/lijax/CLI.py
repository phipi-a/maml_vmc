from functools import partial
from typing import Callable
from jsonargparse import ActionConfigFile, ArgumentParser
from maml_vmc.lijax.logger.Logger import Logger
from maml_vmc.lijax.Trainer import Trainer


def fit(trainer: Trainer, logger: Logger, save_config_fn: Callable):
    with logger.start_run():
        logger.log_params(save_config_fn)
        trainer.start(logger=logger)


def validate(trainer: Trainer, logger: Logger):
    trainer.validate()


def test(trainer: Trainer, logger: Logger):
    trainer.test()


def run(save_config_fn, init):
    command = init["subcommand"]
    init = init[command]
    del init["config"]
    # logger.log_params(flat_config_dict)
    if command == "fit":
        fit(**init, save_config_fn=save_config_fn)
    elif command == "validate":
        validate(**init)
    elif command == "test":
        test(**init)


def CLI():
    sub_parser = ArgumentParser()
    sub_parser.add_argument("--config", action=ActionConfigFile, help="config file")
    sub_parser.add_subclass_arguments(Trainer, "trainer")
    sub_parser.add_subclass_arguments(Logger, "logger")

    parser = ArgumentParser()
    sub = parser.add_subcommands()
    parser.add_argument("--config", action=ActionConfigFile, help="config file")

    # parser.add_instantiator(instantiator=test, class_type=int)
    sub.add_subcommand("fit", sub_parser)
    sub.add_subcommand("validate", sub_parser)
    sub.add_subcommand("test", sub_parser)
    cfg = parser.parse_args()

    save_config_fn = partial(parser.save, cfg=cfg)
    # do not instantiate functions

    init = parser.instantiate_classes(cfg)
    run(save_config_fn, init)
