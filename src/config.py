# Standard library imports
from pathlib import Path

# Paths =========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "planets"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# Constants =====================================
GENERAL_INFO_MIN_SCORE = 0.75
KNOWN_PLANETS = {
    "mercury",
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "pluto",
}
SESSION_TIMEOUT_SECONDS = 300
USE_TIMEOUT_INPUT = False  # True in PowerShell/CMD, False in PyCharm