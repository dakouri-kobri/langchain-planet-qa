# Imports =======================================
from http.client import responses

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_groq import ChatGroq
import dotenv


# Configuration / Setup =========================

def setup_llm():
    dotenv.load_dotenv()

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_retries=2
    )


# Few-shot Examples =============================

def get_examples():
    examples = [
        {
            "input": "Jupiter",
            "output": (
                "Jupiter is the largest planet in the solar system.\n"
                "It is a gas giant primarily composed of hydrogen and helium.\n"
                "It has a Great Red Spot, a massive storm, and at least 79 known moons."
            )
        },
        {
            "input": "Mars",
            "output": (
                "Mars is the fourth planet from the Sun.\n"
                "It has a thin atmosphere composed mainly of carbon dioxide.\n"
                "Notable features include Olympus Mons and Valles Marineris."
            )
        }
    ]

    return examples


# Prompt builder =============================

def build_prompt():
    examples = get_examples()
    example_template = PromptTemplate.from_template(
        "Planet: {input}\nAnswer:\n{output}\n",
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        suffix = (
            "Planet: {question}\n"
            "Answer:\n"
            "Provide key details including:\n"
            "- Physical characteristics (size, composition, atmosphere)\n"
            "- Notable features (rings, moons, surface conditions\n"
            "- Scientific or historical significance\n"
            "- Fun or surprising facts\n"
        ),
        input_variables=["question"]
    )

    return few_shot_prompt


# Main Application Logic ========================

def main():
    llm = setup_llm()
    prompt_template = build_prompt()

    question = input("Enter a planet name: ")

    final_prompt = prompt_template.format(question=question)

    response = llm.invoke(final_prompt)

    print("========= Answer =========\n")
    print(response.content)


# Entry Point ===================================

if __name__ == "__main__":
    main()
