from pathlib import Path


class ProjectPaths:
    ROOT = Path(__file__).parent.parent
    SRC = ROOT / "searchagent"
    EVAL = SRC / "eval"
    EVAL_CONFIG = EVAL / "config"
    DATA = ROOT / "data"
    EVAL_OUTPUT = ROOT / "eval_output"


project_paths = ProjectPaths()
