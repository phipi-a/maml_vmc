from loguru import logger
import sys


def configure_logger():
    logger.remove()

    # only for info not for debug
    def logger_format(color: str):
        return f"<{color}>" + "{time:HH:mm:ss}" + f"</{color}>" + " {message}"

    logger.add(
        sys.stderr,
        format=logger_format("green"),
        filter=lambda record: record["level"].name == "INFO",
        colorize=True,
    )
    logger.add(
        sys.stderr,
        format=logger_format("blue"),
        filter=lambda record: record["level"].name == "DEBUG",
        colorize=True,
    )
    logger.add(
        sys.stderr,
        format=logger_format("red"),
        filter=lambda record: record["level"].name == "ERROR",
        colorize=True,
    )
