# Imports =======================================
from os.path import exists
from pathlib import Path

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

# Main Application Logic ========================

def main() -> None:
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

    chain = prompt | model_with_tools | run_tools

    user_query = input("Ask about planets: ")

    result = chain.invoke({"question": user_query})

    print(result)
    print(chain)


# Entry Point ===================================

if __name__ == "__main__":
    main()
