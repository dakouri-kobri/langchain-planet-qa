# Third-party library imports
from langchain_core.tools import tool

# Local imports
from vectorstore import build_vectorstore

VECTORSTORE = build_vectorstore()


# Tools =========================================

@tool("PlanetDistanceSun")
def planet_distance_sun(planet_name: str) -> str:
    """Return the approximate distance of a planet from the Sun in AU."""
    distances = {
        "earth": "Earth is approximately 1 AU from the Sun.",
        "mars": "Mars is approximately 1.5 AU from the Sun.",
        "jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
        "pluto": "Pluto is approximately 39.5 AU from the Sun.",
    }

    return distances.get(
        planet_name.lower(),
        f"Information about the distance of {planet_name} from the Sun is not available in this tool.",
    )


@tool("PlanetRevolutionPeriod")
def planet_revolution_period(planet_name: str) -> str:
    """Return the approximate revolution period of a planet around the Sun in Earth years."""
    periods = {
        "earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
        "mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
        "jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
        "pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun.",
    }

    return periods.get(
        planet_name.lower(),
        f"Information about the revolution period of {planet_name} is not available in this tool.",
    )


@tool("PlanetGeneralInfo")
def planet_general_info(planet_name: str) -> str:
    """Return general information about a planet using similarity search over the documents."""
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

def run_tools(payload: dict) -> str:
    question = payload["question"].strip().lower()
    message = payload["message"]

    if not getattr(message, "tool_calls", None):
        return "The requested information is not available in our database."

    tool_call = message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Guardrail 1: PlanetDistanceSun is ONLY for distance from the Sun
    if tool_name == "PlanetDistanceSun":
        if "sun" not in question:
            return "Information about distances not involving the Sun is not available in this system."

    # Guardrail 2: PlanetRevolutionPeriod is ONLY for revolving/orbiting around the Sun
    if tool_name == "PlanetRevolutionPeriod":
        if "sun" not in question and "orbit" not in question and "revolve" not in question:
            return "Information about that revolution or orbital relationship is not available in this system."

    selected_tool = TOOL_MAP[tool_name]

    return selected_tool.invoke(tool_args)