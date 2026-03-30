# Third-party library imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from config import CHROMA_DIR
from data_loader import load_documents

# Vector Store Builder ==========================
def build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore: Chroma = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Only build DB if it doesn't exist yet
    if not (CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())):
        docs = load_documents()
        vectorstore.add_texts(docs)

    return vectorstore