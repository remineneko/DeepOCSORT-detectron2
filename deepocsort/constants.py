import os
from pathlib import Path

DEEPOCSORT_ROOT = Path(__file__).parent
WEIGHTS_FOLDER = DEEPOCSORT_ROOT / "weights"

WEIGHTS_FOLDER.mkdir(exist_ok=True)