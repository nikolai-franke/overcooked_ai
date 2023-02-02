import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.join(_current_dir, os.pardir)
RUNS_DIR = os.path.join(_base_dir, "runs")
MODELS_DIR = os.path.join(_base_dir, "models")
GIFS_DIR = os.path.join(_base_dir, "gifs")
