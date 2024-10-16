from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import List, TypeAlias

import numpy as np
from loguru import logger

VectorList: TypeAlias = List[np.array]


class ProjectPaths:
    ROOT = Path(__file__).parent.parent
    SRC = ROOT / "searchagent"
    EVAL = SRC / "eval"
    EVAL_CONFIG = EVAL / "config"
    DATA = ROOT / "data"
    EVAL_OUTPUT = ROOT / "eval_output"
    TEST_SOURCES = DATA / "test_sources"
    PDF_DIR = TEST_SOURCES / "pdfs"
    SINGLE_FILE_DIR = TEST_SOURCES / "single_file_dir"


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


def get_now():
    return datetime.now(timezone.utc).isoformat()


project_paths = ProjectPaths()
