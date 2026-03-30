# Standard Library imports
import msvcrt
import time

# Local imports
from config import SESSION_TIMEOUT_SECONDS, USE_TIMEOUT_INPUT


# Timed Input Helper ============================
def timed_input(prompt: str, timeout_seconds: int) -> str | None:
    print(prompt, end="", flush=True)

    buffer: list[str] = []
    start_time = time.time()

    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwch()

            if char == "\r":
                print()
                return "".join(buffer)

            if char == "\b":
                if buffer:
                    buffer.pop()
                    print("\b \b", end="", flush=True)
                continue

            if char in ("\x00", "\xe0"):
                _ = msvcrt.getwch()
                continue

            buffer.append(char)
            print(char, end="", flush=True)

        if time.time() - start_time >= timeout_seconds:
            print()
            return None

        time.sleep(0.05)

# User Input ====================================
def get_user_input(prompt: str) -> str | None:
    if USE_TIMEOUT_INPUT:
        return timed_input(prompt, SESSION_TIMEOUT_SECONDS)
    return input(prompt)


# Single-Question Logic =========================
def answer_question(chain, question: str) -> str:
    cleaned_question = question.strip()

    if not cleaned_question:
        return "Please enter a question."

    return chain.invoke({"question": cleaned_question})