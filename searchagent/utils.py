from pathlib import Path
from time import time

from loguru import logger


class ProjectPaths:
    ROOT = Path(__file__).parent.parent
    SRC = ROOT / "searchagent"
    EVAL = SRC / "eval"
    EVAL_CONFIG = EVAL / "config"
    DATA = ROOT / "data"
    EVAL_OUTPUT = ROOT / "eval_output"


def timer(func):
    def wrap_function(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()

        runtime = end - start

        unit = "ms" if runtime < 0.001 else "s"
        scaled_time = runtime * 1000 if unit == "ms" else runtime

        logger.info(f"Function {func.__name__!r} executed in {scaled_time:.4f} {unit}")
        return result

    return wrap_function


project_paths = ProjectPaths()
