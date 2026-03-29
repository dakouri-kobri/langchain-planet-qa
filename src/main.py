# Imports =======================================

from pathlib import Path
import dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Paths =========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "planets"


# Load Documents ================================

def load_documents():
    docs = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            docs.append(content)

    return docs


# Build Vector Store ============================

def build_vectorstore(docs: list[str]) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        embedding_function=embeddings
    )

    vectorstore.add_texts(docs)

    return vectorstore


# Main Application Logic ========================

def main() -> None:
    dotenv.load_dotenv()

    docs = load_documents()
    vectorstore = build_vectorstore(docs)

    query = input("Ask a question about planets: ")

    results = vectorstore.similarity_search(query, k=1)

    print("========= Most Relevant Document =========\n")
    print(results[0].page_content)


# Entry Point ===================================

if __name__ == "__main__":
    main()
