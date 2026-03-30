# 🌌 Introduction to LangChain — Planet QA System

A modular **LLM-powered question-answering system** built with LangChain.
This project demonstrates how to combine **prompt engineering, vector search, and tool usage** to create a robust and controlled AI application.

---

## 🚀 Features

* 🔹 Prompt-based LLM interaction using LangChain
* 🔹 Retrieval-Augmented Generation (RAG) with Chroma vector database
* 🔹 Custom tools for structured reasoning:

  * Planet distance from the Sun
  * Planet revolution period
  * General planetary information via embeddings
* 🔹 Guardrails to prevent hallucinations
* 🔹 Modular architecture (clean separation of concerns)
* 🔹 Interactive CLI session with optional inactivity timeout
* 🔹 Persistent vector store (no re-indexing on each run)

---

## 🧠 Project Architecture

```
src/
├── main.py              # Entry point
├── config.py            # Configuration and constants
├── data_loader.py       # Load planet text data
├── vectorstore.py       # Embeddings + Chroma DB
├── tools.py             # Custom tools + validation logic
├── chain_builder.py     # LLM + prompt + runnable chain
└── session.py           # CLI session + input handling
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd IntroductionToLangChain
```

### 2. Install dependencies (using `uv`)

```bash
uv venv
uv pip install -r requirements.txt
```

> Or install manually:

```bash
uv pip install langchain langchain-core langchain-groq langchain-chroma langchain-huggingface python-dotenv sentence-transformers
```

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

> ⚠️ Make sure `.env` is included in `.gitignore`

---

## ▶️ Running the Application

```bash
python src/main.py
```

---

## 💬 Example Usage

```
Ask about planets: How far is Pluto from the Sun?
→ Pluto is approximately 39.5 AU from the Sun.

Ask about planets: Tell me about Earth.
→ Earth is the third planet from the Sun and the only known planet to support life...

Ask about planets: How far is Pluto to Earth?
→ Information about distances not involving the Sun is not available in this system.

Ask about planets: Tell me about planet Zeor.
→ Additional information for Zeor is not available in this tool.
```

---

## 🛡️ Guardrails & Reliability

This system avoids hallucinations through:

* ✅ Tool-based architecture (LLM must use tools)
* ✅ Input validation before tool execution
* ✅ Known-planet whitelist
* ✅ Vector similarity threshold filtering
* ✅ Explicit fallback responses for unsupported queries

---

## 🧩 Technologies Used

* **LangChain** — LLM orchestration
* **Groq API** — fast LLM inference
* **ChromaDB** — vector storage
* **HuggingFace Embeddings** — semantic search
* **Python 3.11**
* **uv** — modern Python package manager

---

## 📚 Learning Objectives

This project demonstrates:

* Prompt engineering with templates
* Few-shot prompting
* Vector search and embeddings
* Tool calling in LLMs
* Runnable pipelines (LangChain Expression Language)
* Building modular AI applications

---

## ⚠️ Known Limitations

* Limited to predefined planetary dataset
* CLI-based interface (no UI yet)
* Basic rule-based guardrails (not full semantic validation)

---

## 🔮 Future Improvements

* 🌐 Web interface (Streamlit / FastAPI)
* 🧠 Better intent classification
* 📊 Logging and observability
* 🗂️ Support for larger datasets
* 🤖 Multi-step reasoning agents

---

## 👨‍💻 Author

Dakouri Kobri<br>
Health Science, Data Science, & AI Enthusiast

---

## 📄 License

This project is for educational purposes.
