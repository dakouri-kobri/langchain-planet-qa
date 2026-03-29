# Imports =======================================

from langchain_core.prompts import PromptTemplate
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


# Prompt definition =============================

def build_prompt(question:str):
    template = PromptTemplate.from_template(
        "You are a helpful assistant who answers questions users may have. "
        "You are asked: {question}."
    )

    return template.invoke({"question": question})


# Main Application Logic ========================

def main():
    llm = setup_llm()
    question = input("What is your question? ")
    prompt = build_prompt(question)
    response = llm.invoke(prompt)

    print(response.content)


# Entry Point ===================================

if __name__ == "__main__":
    main()
