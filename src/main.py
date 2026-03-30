# Local imports =================================
from chain_builder import build_chain
from config import USE_TIMEOUT_INPUT
from session import answer_question, get_user_input


# Main Application Logic ========================
def main() -> None:
    chain = build_chain()

    print("Planet QA session started.")
    print("Type your question, or type 'exit' or 'q' to quit.")

    if USE_TIMEOUT_INPUT:
        print("Session closes after 5 minutes of inactivity.\n")
    else:
        print("Inactivity timeout is disabled in this console.\n")

    while True:
        user_query = get_user_input("Ask about planets: ")

        if user_query is None:
            print("\nSession closed due to inactivity.")
            break

        if user_query.strip().lower() in {"exit", "quit", "q"}:
            print("\nSession closed by user.")
            break

        result = answer_question(chain, user_query)
        print(f"\n{result}\n")


# Entry Point ===================================
if __name__ == "__main__":
    main()
