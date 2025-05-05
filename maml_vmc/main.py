import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["JAX_DISABLE_JIT"] = "false"
os.environ["JAX_DEBUG_NANS"] = "false"
# prealocate 95 % of the memory
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# check if fit in sys.argv
# if len(sys.argv) > 1 and sys.argv[1] == "fit":
#     os.environ["JAX_ENABLE_X64"] = "false"
# else:
#     os.environ["JAX_ENABLE_X64"] = "true"
from maml_vmc import lijax
from maml_vmc.utils.Logger import configure_logger


if __name__ == "__main__":
    configure_logger()
    lijax.CLI()
