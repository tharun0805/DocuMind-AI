🧠DocuMind AI

**Intelligent Document Intelligence Platform**

Upload any document. Ask anything in plain English. Get instant AI-powered answers, charts, quizzes, and knowledge — all private.

**Live Demo:** https://documind-ai.streamlit.app

---

## What Is DocuMind AI?

DocuMind AI lets you upload documents and have a conversation with them. Instead of manually reading through pages of content, you simply ask a question and get a precise, grounded answer in seconds.

It uses a multi-agent AI system that intelligently routes your question to the right tool — whether that is semantic search, keyword search, data computation, or a combination of all three.

---

## Features

**Document Understanding**
- Upload PDF, DOCX, PPTX, XLSX, CSV, or TXT files
- Supports multiple documents simultaneously
- Extracts text, tables, and structured data

**AI Capabilities**
- Answers questions using hybrid search — semantic + keyword combined
- Computes answers from Excel and CSV using a DataFrame agent
- Generates charts and graphs from document data
- Creates multiple choice quizzes based on document content
- Extracts entities, insights, action items, and key metrics
- Summarizes documents in multiple styles — detailed, bullet points, beginner-friendly, executive, and more

**On Demand Features** (only shown when you ask for them)
- Charts — say "show me a chart"
- Quiz — say "give me a quiz"
- Resources — say "show resources" for YouTube and web links
- Export — say "export as PDF" to download your answer

**Other Features**
- Read Aloud — browser text-to-speech on every answer
- Smart Prompts — auto-generated questions from your document
- Stop Button — cancel a query mid-processing
- Multi-document query and comparison
- Session memory — remembers conversation context

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Interface | Streamlit |
| Primary LLM | Groq LLaMA 3.3 70B |
| Fallback LLM | Google Gemini 2.5 Flash |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Retrieval Strategy | Hybrid with Reciprocal Rank Fusion |
| Agent Framework | LangGraph |
| LLM Framework | LangChain |
| Logging | Loguru |
| Containerization | Docker |

---

## Prerequisites

Before you begin, make sure you have:

- Python 3.9 or higher
- Git
- A free Groq API key — get one at https://console.groq.com
- A free Google API key — get one at https://aistudio.google.com/app/apikey

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/tharun0805/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a virtual environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac or Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your environment file

```bash
cp .env.example .env
```

Open the `.env` file and add your API keys:

```
GOOGLE_API_KEY=paste_your_google_key_here
GROQ_API_KEY=paste_your_groq_key_here
```

### 5. Cache the embedding model (recommended)

This downloads the AI model once so the app starts faster every time after.

```bash
python -c "
from sentence_transformers import SentenceTransformer
import os
os.makedirs('models/embedding_model', exist_ok=True)
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
m.save('models/embedding_model')
print('Done — model saved locally')
"
```

### 6. Run the app

**Windows:**
```powershell
$env:PYTHONIOENCODING = "utf-8"
streamlit run app/main.py --server.fileWatcherType none
```

**Mac or Linux:**
```bash
PYTHONIOENCODING=utf-8 streamlit run app/main.py --server.fileWatcherType none
```

Open your browser and go to: **http://localhost:8501**

---

## Docker Setup

If you prefer to run the app inside Docker:

### 1. Make sure Docker is installed

Download Docker Desktop from https://www.docker.com/products/docker-desktop

### 2. Build and run

```bash
docker-compose up --build
```

Open your browser and go to: **http://localhost:8501**

### Stop the app

```bash
docker-compose down
```

---

## Streamlit Cloud Deployment

To deploy the app publicly for free:

**Step 1** — Push your code to GitHub

**Step 2** — Go to https://share.streamlit.io and sign in with GitHub

**Step 3** — Click **New App** and fill in:
- Repository: `tharun0805/DocuMind-AI`
- Branch: `main`
- Main file path: `app/main.py`

**Step 4** — Click **Advanced settings** then click **Secrets**

Paste this into the secrets box:
```toml
GOOGLE_API_KEY = "your_actual_google_key"
GROQ_API_KEY   = "your_actual_groq_key"
```

**Step 5** — Click **Deploy**

Your app will be live at a URL like `https://documind-ai.streamlit.app` in about 3 minutes.

**Note:** Voice input works automatically on the deployed URL because it uses HTTPS.

---

## How to Use the App

### Uploading a document
1. Use the sidebar to upload your file
2. Click **Process Document**
3. Wait 5-10 seconds for processing
4. Smart questions appear automatically — click any to get started

### Asking questions
Type your question in the chat box at the bottom and press Enter.

### Special requests you can make

| What you type | What happens |
|--------------|-------------|
| `show me a chart` | Generates a visual chart from document data |
| `give me a quiz` | Creates 5 multiple choice questions |
| `show resources` | Gets YouTube and web links on the topic |
| `export as pdf` | Downloads your answer as a PDF file |
| `export as excel` | Extracts data to an Excel spreadsheet |
| `create a presentation` | Generates a PowerPoint from the answer |
| `summarize the document` | Full document summary |
| `extract action items` | Lists all tasks and actions |
| `identify risks` | Lists all risks mentioned |

### Working with multiple documents
1. Process your first document
2. Upload a second file using the **Add Document** button in the sidebar
3. Use the **Multi-Document Panel** that appears to:
   - Ask a question across all documents at once
   - Compare all documents side by side

---

## Project Structure

```
DocuMind-AI/
├── app/
│   └── main.py                 Main application file
├── agents/
│   ├── intent_agent.py         Classifies what the user wants
│   ├── planner_agent.py        Decides which tools to use
│   ├── qa_agent.py             Generates the final answer
│   ├── retriever_agent.py      Fetches relevant document chunks
│   ├── dataframe_agent.py      Handles Excel and CSV questions
│   └── multi_document_agent.py Queries across multiple documents
├── chunking/
│   └── text_chunker.py         Splits documents into chunks
├── embeddings/
│   └── embedding_model.py      Loads the embedding model
├── graph/
│   └── workflow.py             LangGraph agent workflow
├── ingestion/
│   ├── document_loader.py      Routes files to the right loader
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── pptx_loader.py
│   ├── xlsx_loader.py
│   ├── csv_loader.py
│   └── txt_loader.py
├── memory/
│   ├── session_memory.py       Conversation memory
│   └── file_memory_manager.py  Per-document memory
├── retrieval/
│   └── hybrid_retriever.py     FAISS + BM25 + RRF
├── tools/
│   ├── dataframe_tool.py       Pandas computation
│   └── file_export_tool.py     PDF, DOCX, XLSX export
├── utils/
│   ├── config.py               API key management
│   ├── error_handler.py        Error handling
│   ├── logger.py               Structured logging
│   ├── llm_provider.py         LLM with retry logic
│   ├── performance.py          Timing tracker
│   ├── security_audit.py       Secret scanner
│   └── validator.py            Input validation
├── vector_store/
│   ├── faiss_store.py          FAISS index
│   └── bm25_store.py           BM25 index
├── .env.example                API key template
├── .gitignore
├── .streamlit/config.toml      Streamlit settings
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Environment Variables

| Variable | Required | Where to get it |
|----------|----------|----------------|
| `GROQ_API_KEY` | Recommended | https://console.groq.com |
| `GOOGLE_API_KEY` | Required | https://aistudio.google.com/app/apikey |

Groq is recommended because it is 10x faster than Gemini on the free tier. If Groq is not set, the app automatically uses Gemini as a fallback.

---

## Security

- API keys are stored only in the `.env` file — never in the source code
- The `.env` file is listed in `.gitignore` and will never be pushed to GitHub
- All user inputs are validated and sanitized before processing
- A security scanner checks all source files for accidentally hardcoded secrets on startup

---

## Common Issues

**App starts slowly on first run**

The embedding model downloads on first run (~90MB). Run the cache command in Step 5 above to fix this permanently.

**Answers say "Result: None"**

This means the workflow routed to the wrong agent. The app will automatically fall back to direct summarization. Try rephrasing your question.

**Charts are not generating**

Make sure your document contains numerical data. Try being specific: "show me the scores as a bar chart" works better than "show a chart".

**Voice input not working**

Voice input requires HTTPS. It works automatically on the Streamlit Cloud deployment URL. It will not work on localhost.

---

## License

MIT License — see the LICENSE file for details.

---

## Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Multi-agent workflow
- [Groq](https://groq.com) — Fast LLM inference
- [Google Gemini](https://deepmind.google/technologies/gemini/) — Fallback LLM
- [FAISS](https://faiss.ai) — Vector similarity search
- [Streamlit](https://streamlit.io) — Web app framework
- [HuggingFace](https://huggingface.co) — Embedding models

---

*DocuMind AI — Built from scratch. Built to last.*
