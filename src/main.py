from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import dotenv


def main():
    dotenv.load_dotenv()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_retries=2,
    )

    question = input("Ask something: ")

    template = PromptTemplate.from_template(
        "You are a helpful assistant who answers user questions. "
        "You are asked: {question}"
    )

    prompt = template.invoke({"question": question})

    response = llm.invoke(prompt)

    print(response.content)


if __name__ == "__main__":
    main()