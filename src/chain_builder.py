# Third-party library imports
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

# Local imports
from tools import TOOLS, run_tools

# Chain Builder =================================
def build_chain():
    dotenv.load_dotenv()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_retries=2,
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant that must answer planet questions using only the available tools.\n"
        "Use PlanetDistanceSun only for questions about a planet's distance from the Sun.\n"
        "Use PlanetRevolutionPeriod only for questions about how long a planet takes to orbit or revolve around the Sun.\n"
        "Use PlanetGeneralInfo for general factual information about a planet from the data source.\n"
        "If the question asks for unsupported information, such as distance between two planets or a fact not available in the data source, do not guess.\n"
        "Select the most appropriate tool or return no tool call.\n"
        "Question: {question}"
    )

    model_with_tools = llm.bind_tools(TOOLS)

    chain = (
        RunnablePassthrough.assign(
            prompt_value=prompt
        )
        | RunnablePassthrough.assign(
            message=RunnableLambda(
                lambda x: model_with_tools.invoke(x["prompt_value"])
            )
        )
        | RunnableLambda(run_tools)
    )

    return chain