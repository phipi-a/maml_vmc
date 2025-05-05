import os
from typing import Literal, NamedTuple
from jsonargparse import ActionConfigFile, ArgumentParser
import numpy as np
import itertools
from tqdm import tqdm


class Param(NamedTuple):
    name: str
    values: list | tuple
    type: Literal["categorical", "range"]
    num_values: int | None = None


def get_configurations(params: list[Param]):
    list_of_values = []
    for param in params:
        values = param.values
        if param.type == "range":
            values = list(
                np.linspace(param.values[0], param.values[1], param.num_values)
            )
        list_of_values.append(values)

    v = list(itertools.product(*list_of_values))
    return v


def sweep(command: str, params: list[Param], type: Literal["grid", "random"]):
    v = get_configurations(params)
    if type == "random":
        np.random.shuffle(v)
    print(f"Running {len(v)} configurations")
    # ask for confirmation
    if input("Do you want to continue? [y/n]: ").lower() != "y":
        return
    for i, values in tqdm(enumerate(v)):
        final_command = command
        for param, value in zip(params, values):
            final_command += f" --{param.name} {value}"
        final_command += " --logger.run_name sweep_" + str(i)
        print(f"Running command ({i+1}/{len(v)}): {final_command}")
        os.system(final_command)


parser = ArgumentParser()
parser.add_argument("--config", action=ActionConfigFile, help="config file")

parser.add_function_arguments(sweep)
cfg = parser.parse_args()
# initialize the classes
init = parser.instantiate_classes(cfg)

del init["config"]
sweep(**init)
