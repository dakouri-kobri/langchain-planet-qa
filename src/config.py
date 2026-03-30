# Standard library imports
from pathlib import Path

# Paths =========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "planets"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# Constants =====================================
SESSION_TIMEOUT_SECONDS = 300
USE_TIMEOUT_INPUT = False  # True in PowerShell/CMD, False in PyCharm