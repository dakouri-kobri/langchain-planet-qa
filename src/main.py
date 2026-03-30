# Imports =======================================

from pathlib import Path
from queue import Queue, Empty
from threading import Thread

import dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Paths =========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "planets"
CHROMA_DIR = ROOT_DIR / "chroma_db"


# Load Documents ================================

def load_documents() -> list[str]:
    docs = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            docs.append(content)

    return docs


# Build Vector Store ============================

def build_vectorstore() -> Chroma:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore: Chroma = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    # noinspection PyProtectedMember
    existing_count = vectorstore._collection.count()

    if existing_count == 0:
        docs = load_documents()
        vectorstore.add_texts(docs)

    return vectorstore

VECTORSTORE = build_vectorstore()


# Tools =========================================

@tool("PlanetDistanceSun")
def planet_distance_sun(planet_name: str) -> str:
    """Returns approximate distance of a planet from the sun in AU."""

    distances = {
        "earth": "Earth is approximately 1 AU from the Sun.",
        "mars": "Mars is approximately 1.5 AU from the Sun.",
        "jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
        "pluto": "Pluto is approximately 39.5 AU from the Sun.",
    }

    return distances.get(
        planet_name.lower(),
        f"Information about the distance of {planet_name} from the Sun is not available in this tool."
    )

@tool("PlanetRevolutionPeriod")
def planet_revolution_period(planet_name: str) -> str:
    """Returns approximate revolution period of a planet in Earth years."""

    periods = {
        "earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
        "mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
        "jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
        "pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun.",
    }

    return periods.get(
        planet_name.lower(),
        f"Information about the revolution period of {planet_name} is not available in this tool."
    )


@tool("PlanetGeneralInfo")
def planet_general_info(planet_name: str) -> str:
    """Return general information about a planet using similarity search over planet documents."""

    results = VECTORSTORE.similarity_search(planet_name, k=1)

    if results:
        return results[0].page_content

    return f"Additional information for {planet_name} is not available in this tool."

TOOLS = [
    planet_distance_sun,
    planet_revolution_period,
    planet_general_info,
]

TOOL_MAP = {
    "PlanetDistanceSun": planet_distance_sun,
    "PlanetRevolutionPeriod": planet_revolution_period,
    "PlanetGeneralInfo": planet_general_info,
}


# Tool Execution Logic ==========================

def run_tools(message) -> str:
    if not getattr(message, "tool_calls", None):
        return "No suitable tool was selected."

    tool_call = message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    selected_tool = TOOL_MAP[tool_name]

    return selected_tool.invoke(tool_args)

# Timed Input Helper ============================

def timed_input(prompt: str, timeout_seconds: int = 300) -> str | None:
    queue: Queue[str] = Queue()

    def reader() -> None:
        try:
            user_text = input(prompt)
            queue.put(user_text)
        except EOFError:
            queue.put("exit")

    thread = Thread(target=reader, daemon=True)
    thread.start()

    try:
        return queue.get(timeout=timeout_seconds)
    except Empty:
        return None


# Chain Builder =================================

def build_chain():
    dotenv.load_dotenv()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_retries=2,
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant who answers questions users may have. "
        "You are asked: {question}."
    )

    model_with_tools = llm.bind_tools(TOOLS)

    return prompt | model_with_tools | run_tools


# Single-Question Logic =========================

def answer_question(chain, question: str) -> str:
    cleaned_question = question.strip()

    if not cleaned_question:
        return "Please enter a question."

    return chain.invoke({"question": cleaned_question})


# Main Application Logic ========================

def main() -> None:
    chain = build_chain()

    print("Planet QA session started.")
    print("Type your question, or type 'exit' or 'q' to quit.")
    print("Session closes after 5 minutes of inactivity.\n")

    while True:
        user_query = timed_input("Ask about planets: ")

        if user_query is None:
            print("\nSession closed due to inactivity.")
            break

        if user_query.strip().lower() in {"exit", "quit", "q"}:
            print("\nSession closed by user.")
            break

        result = answer_question(chain, user_query)
        print("\n" + result + "\n")


# Entry Point ===================================

if __name__ == "__main__":
    main()
