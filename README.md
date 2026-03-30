# 🌌 Introduction to LangChain<br><span> —  _Building a Planet QA System_

A modular LLM-powered question-answering system that answers questions about planets using LangChain, Chroma vector search, Groq models, and Hugging Face embeddings.

This project demonstrates how to build a reliable AI application by combining:

- prompt engineering
- retrieval-augmented generation (RAG)
- tool-based reasoning
- runnable pipelines
- guardrails to prevent hallucinations

---

## 🚀 Features

- Prompt-based question answering with LangChain
- Retrieval of planetary information using Chroma vector database
- Custom tools for:
  - distance from the Sun
  - revolution period
  - general planet information
- Runnable chain composition with LangChain
- Guardrails to avoid unsupported answers
- Persistent vector store
- Interactive CLI session

---
## 🧩 Tech Stack

* **Python 3.11**
* **LangChain** — LLM orchestration
* **Groq API** — fast LLM inference
* **ChromaDB** — vector storage
* **HuggingFace Embeddings** — semantic search
* **uv** — modern Python package manager

---

## 📁 Project Structure

```
data/
└── planets/              # txt files, source of primary infomation on planets

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
git clone https://github.com/dakouri-kobri/langchain-planet-qa.git
cd IntroductionToLangChain
```

### 2. Install dependencies (using `uv`)

```bash
uv venv
.venv\Scripts\activate
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

> ⚠️ Make sure `.env` is included in `.gitignore` — do not commit your API keys.

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

This system includes validation to reduce hallucinations through:

* ✅ Tool-based architecture (LLM must use tools)
* ✅ Input validation before tool execution
* ✅ Known-planet whitelist
* ✅ Vector similarity threshold filtering
* ✅ Explicit fallback responses for unsupported queries

---

⚠️ Limitations
- Limited to the provided planetary dataset
- CLI-only interface
- Rule-based validation rather than full semantic verification

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

## 🔮 Future Improvements

* 🌐 Web interface (Streamlit / FastAPI)
* 🧠 Better query intent classification
* 📊 Logging and observability
* 🗂️ Support for larger datasets
* 🤖 Multi-step reasoning agents

---

## 👨‍💻 Author

Dakouri Kobri<br>
Health Science, Data Science, & AI Enthusiast

---

## 📄 License

This project is for educational purposes. It is licensed under the terms of the LICENSE file in the repository.
