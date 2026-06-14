# 📰 AI News Research Assistant

![Python](https://img.shields.io/badge/Python-3.x-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![HuggingFace](https://img.shields.io/badge/LLM-Qwen2.5--7B-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Working-success)

An AI-powered News Research Assistant built using LangChain, FAISS, HuggingFace Models, and Streamlit.

The system allows users to provide news article URLs, automatically builds a semantic knowledge base from those articles, and answers questions using Retrieval-Augmented Generation (RAG).

The project demonstrates both:

* Stuff Method RAG
* Map-Reduce RAG

for handling long-form news articles efficiently.

---

# 🚀 Features

✅ URL-based Knowledge Ingestion

✅ News Article Processing

✅ Automatic Text Chunking

✅ FAISS Vector Database

✅ Semantic Search

✅ MMR Retrieval

✅ Retrieval-Augmented Generation (RAG)

✅ Map-Reduce Summarization Pipeline

✅ Source Attribution

✅ Streamlit Interface

✅ HuggingFace Embeddings

✅ Qwen 2.5 Integration

✅ Multi-Document Question Answering

---

# 📂 Project Structure

```text
news-research-bot/
│
├── app.py
│
├── faiss_store/
│   ├── index.faiss
│   └── index.pkl
│
├── notebooks/
│   ├── URL_BOT(stuff_method).ipynb
│   └── URL_BOT(map-reduce_method).ipynb
│
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 🏗 Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
URL Loader
 │
 ▼
Text Splitter
 │
 ▼
Embeddings
 │
 ▼
FAISS Vector Store
 │
 ▼
MMR Retriever
 │
 ▼
Map Step
 │
 ▼
Reduce Step
 │
 ▼
Final Answer
```

The application follows a Retrieval-Augmented Generation architecture:

* UI Layer → Streamlit
* Retrieval Layer → LangChain Retriever
* Vector Database Layer → FAISS
* Embedding Layer → HuggingFace Embeddings
* LLM Layer → Qwen 2.5 7B
* Data Layer → News URLs

---


# ⚙️ How It Works

### Step 1

User enters one or more news article URLs.

Example:

```text
https://news-site.com/article1
https://news-site.com/article2
```

---

### Step 2

The articles are loaded using:

```python
UnstructuredURLLoader
```

---

### Step 3

Articles are split into chunks using:

```python
RecursiveCharacterTextSplitter
```

---

### Step 4

Embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

### Step 5

Embeddings are stored in:

```text
FAISS
```

---

### Step 6

Relevant chunks are retrieved using:

```python
search_type="mmr"
```

---

### Step 7

Map Stage

Each retrieved chunk is independently filtered and summarized.

---

### Step 8

Reduce Stage

Filtered chunks are merged into a final context and sent to the LLM.

---

### Step 9

The assistant generates the final grounded answer with source references.

---

# 🔍 Retrieval Pipeline

```text
Question
   │
   ▼
MMR Retrieval
   │
   ▼
Top Chunks
   │
   ▼
Map Prompt
   │
   ▼
Relevant Information
   │
   ▼
Reduce Prompt
   │
   ▼
Final Answer
```

---

# 💬 Example Questions

```text
What did the CEO announce?

Summarize the article.

What are the key financial results?

What happened in the election?

What are the main risks discussed?

Which companies were mentioned?
```

---

# 🔑 Environment Setup

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
```

---

# ⚙️ Installation

# Dependency Management

This project uses:

- pyproject.toml
- uv.lock

Install all dependencies with:

```bash
uv sync

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/news-research-bot.git

cd news-research-bot
```

Install dependencies:

```bash
uv sync
```

---

# 🌐 Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 📸 Screenshots

## URL Input

![URL Input](assets/url-input.png)

---

## Research Interface

![Research Interface](assets/homepage.png)

---

## Answer Generation

![Answer Example](assets/answer-example.png)

---

# 🧠 Retrieval Strategy

The application uses:

```python
search_type="mmr"
```

with:

```python
k=8
fetch_k=20
```

Advantages:

* Better document diversity
* Reduced redundancy
* Improved answer quality
* More relevant context retrieval

---

# 📦 Tech Stack

### Programming Language

* Python

### Frameworks

* LangChain
* Streamlit

### LLM

* Qwen 2.5 7B Instruct

### Embeddings

* all-MiniLM-L6-v2

### Vector Database

* FAISS

### Retrieval

* MMR Retrieval

### RAG Methods

* Stuff Method
* Map-Reduce Method

---

# 📦 Requirements

```text
streamlit

langchain
langchain-core
langchain-community

langchain-huggingface

faiss-cpu

sentence-transformers

transformers

torch

python-dotenv

unstructured

beautifulsoup4
```

Install:

```bash
uv sync
```

---

# 🧪 Testing Performed

✔ URL Loading

✔ Article Chunking

✔ Embedding Generation

✔ FAISS Storage

✔ MMR Retrieval

✔ Map-Reduce Pipeline

✔ Source Attribution

✔ Streamlit Interface

✔ Multi-Document QA

---

# 👨‍💻 Development Notes

This project was developed to demonstrate practical Retrieval-Augmented Generation techniques for real-world research workflows.

The assistant can transform multiple news articles into a searchable knowledge base and answer questions using context retrieved from those articles rather than relying solely on model memory.

The project demonstrates:

* RAG Pipelines
* Semantic Search
* Vector Databases
* MMR Retrieval
* Map-Reduce Summarization
* Prompt Engineering
* Streamlit Applications

---

# 📄 License

This project was developed for educational, portfolio, and internship evaluation purposes.
