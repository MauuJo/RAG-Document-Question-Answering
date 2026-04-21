# 🧠 Advanced Agentic RAG Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Optimized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Cohere](https://img.shields.io/badge/LLM-Cohere-3959A4?style=for-the-badge&logo=cohere&logoColor=white)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

<br/>

> An advanced **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents and ask intelligent, context-aware questions — powered by an **Agentic routing system**, **Hybrid Search**, and **LLM-powered reasoning**.

<br/>

**[🚀 Live Demo](https://rag-document-question-answering-5xt6uvzd9xfcoffupyrzay.streamlit.app/)** · **[🐳 Docker Hub](https://hub.docker.com/r/mauujo/rag-document-question-answering)** · **[🐛 Report Bug](https://github.com/MauuJo/RAG-Document-Question-Answering/issues)**

</div>

---

## 📸 Screenshots

<div align="center">

| App UI                            | Document Upload                   | Chat Interface                  |
| --------------------------------- | --------------------------------- | ------------------------------- |
| ![App UI](./docs/screenshot1.png) | ![Upload](./docs/screenshot2.png) | ![Chat](./docs/screenshot3.png) |

</div>

---

## ✨ Features

| Feature                       | Description                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| 🤖 **Agentic Intent Routing** | Automatically classifies queries as `CHAT` or `RAG` and routes accordingly |
| 🔍 **Hybrid Search**          | Combines dense (semantic) + sparse (keyword) retrieval via Pinecone        |
| 🔁 **Multi-Query Expansion**  | Generates multiple query variants to improve retrieval recall              |
| ⚡ **Streaming Responses**    | Token-by-token streaming for a fast, interactive experience                |
| 📄 **PDF Question Answering** | Upload any PDF and ask questions grounded in its content                   |
| 🐳 **Dockerized Deployment**  | One-command deployment with Docker                                         |
| 📱 **Mobile-Friendly UI**     | Responsive Streamlit interface that works on any device                    |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Agent Router      │  ──── Classifies intent: CHAT or RAG
└─────────────────────┘
    │               │
    ▼               ▼
  CHAT             RAG
    │               │
    │          ┌────┴─────────────┐
    │          │ Query Expansion  │  ──── Generates N query variants
    │          └────┬─────────────┘
    │               │
    │          ┌────▼─────────────┐
    │          │  Hybrid Search   │  ──── Dense + Sparse via Pinecone
    │          └────┬─────────────┘
    │               │
    │          ┌────▼─────────────┐
    │          │ Context Retrieval│  ──── Top-K relevant chunks
    │          └────┬─────────────┘
    │               │
    ▼               ▼
┌─────────────────────────────┐
│   Cohere LLM (command-a)    │  ──── Generates grounded answer
└─────────────────────────────┘
    │
    ▼
Streaming Response to User
```

---

## 🧰 Tech Stack

| Layer                | Technology                                          |
| -------------------- | --------------------------------------------------- |
| **Frontend**         | [Streamlit](https://streamlit.io/)                  |
| **LLM**              | [Cohere](https://cohere.com/) — `command-a-03-2025` |
| **Vector Database**  | [Pinecone](https://www.pinecone.io/)                |
| **Embeddings**       | Cohere Embeddings + Sentence Transformers           |
| **Document Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io/)          |
| **Orchestration**    | [LangChain](https://www.langchain.com/)             |
| **Containerization** | [Docker](https://www.docker.com/)                   |

---

## 🚀 Getting Started

### Option 1 — Docker (Recommended)

The fastest way to get up and running.

**Step 1: Pull the image**

```bash
docker pull mauujo/rag-document-question-answering:1.0
```

**Step 2: Create a `.env` file**

```env
ACCESS_PASSWORD=your_password
COHERE_API_KEY=your_cohere_key
PINECONE_API_KEY=your_pinecone_key
```

**Step 3: Run the container**

```bash
docker run -p 8501:8501 --env-file .env mauujo/rag-document-question-answering:1.0
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser. ✅

---

### Option 2 — Local Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/MauuJo/RAG-Document-Question-Answering.git
cd RAG-Document-Question-Answering
```

**Step 2: Create and activate a virtual environment**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Create a `.env` file in the project root**

```env
ACCESS_PASSWORD=your_password
COHERE_API_KEY=your_cohere_key
PINECONE_API_KEY=your_pinecone_key
```

**Step 5: Launch the app**

```bash
streamlit run entrypoint/app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser. ✅

---

### Option 3 — Streamlit Cloud

1. Fork this repository and connect it to [Streamlit Cloud](https://streamlit.io/cloud).
2. Add the following **Secrets** in the Streamlit dashboard:

```toml
COHERE_API_KEY = "..."
PINECONE_API_KEY = "..."
INTERVIEW_PASSWORD = "..."
```

3. Click **Deploy** — that's it!

---

## 🧭 How to Use the App

Once the app is running:

1. **Get your API keys:**
   - Cohere → [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)
   - Pinecone → [app.pinecone.io](https://app.pinecone.io/)

2. **Enter your API keys** in the sidebar of the app UI.

3. **Upload a PDF document** using the file uploader.

4. Click **"Process Document"** to index the content into Pinecone.

5. Click **"Initialize Chat"** to prepare the conversation engine.

6. **Start asking questions** about your document!

---

## 📂 Project Structure

```
RAG-Document-Question-Answering/
│
├── entrypoint/
│   └── app.py               # Streamlit app entry point
│
├── src/
│   ├── chatbot.py           # Agentic router + LLM response logic
│   ├── retriever.py         # Hybrid search + query expansion
│   └── ingestion.py         # PDF parsing + Pinecone indexing
│
├── docs/                    # Screenshots and documentation assets
│
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── packages.txt             # System-level packages for deployment
└── README.md
```

---

## 🔐 Environment Variables

| Variable                                 | Required | Description                              |
| ---------------------------------------- | -------- | ---------------------------------------- |
| `COHERE_API_KEY`                         | ✅ Yes   | Your Cohere API key for LLM + embeddings |
| `PINECONE_API_KEY`                       | ✅ Yes   | Your Pinecone API key for vector storage |
| `ACCESS_PASSWORD` / `INTERVIEW_PASSWORD` | ✅ Yes   | Password to gate access to the app UI    |

---

## 🛣️ Roadmap

- [ ] 🧠 Persistent chat memory across sessions
- [ ] 📚 Multi-document support with namespace isolation
- [ ] 🔗 Source citations with page-level references
- [ ] 🔐 Full authentication system (OAuth / SSO)
- [ ] 🌐 Support for web URLs as knowledge sources
- [ ] 📊 Analytics dashboard for query insights

---

## 👩‍💻 Author

**Mansi Yadav**

If you found this project useful, consider giving it a ⭐ — it helps a lot!

[![GitHub Stars](https://img.shields.io/github/stars/MauuJo/RAG-Document-Question-Answering?style=social)](https://github.com/MauuJo/RAG-Document-Question-Answering)

---

## 📄 License

This project is open source. See the [LICENSE](./LICENSE) file for details.
