# Local imports
from config import DATA_DIR

# Documents Loader ==============================
def load_documents() -> list[str]:
    docs = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            docs.append(content)

    return docs