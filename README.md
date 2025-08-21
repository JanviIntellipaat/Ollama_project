# Ollama_AI_Janvi — RAG App (DuckDB + Local Vector Store + Streamlit)

A lightweight RAG stack that runs fully offline with **Ollama** for both chat and embeddings, a **file-backed local vector store** (no SQLite/Chroma server required), and a **DuckDB** structured store for tabular/XML data. Includes a simple **Streamlit UI** and Python APIs.


## ✨ Features

- **Local embeddings via Ollama** (`nomic-embed-text` by default).
- **Unstructured store**: file-backed vectors (`*.npy` + `*.jsonl`) with optional HNSW index (you can skip it).
- **Structured store**: DuckDB database for CSV/XLSX/JSON/XML with per-file materialized tables.
- **Per-document deletion** in **both** stores:
  - `FreeVectorDatabase.delete_document(document_id)`
  - `StructuredDataStore.delete_file(file_id)`
- Simple **Streamlit** app for testing and admin tasks.
- Works on a minimal VM (e.g., AlmaLinux) with **Ollama 0.11.4**.


## 🧩 Project Layout (your 5 core files)

```
document_processor.py    # chunking, parsing, and metadata extraction for unstructured docs
free_rag_system.py       # main orchestrator (wraps vector + structured stores)
streamlit_app.py         # Streamlit UI
structured_store.py      # DuckDB-based structured store
vector_database.py       # local file-backed vector store (Ollama embeddings)
```

> You also received fixed variants that add robust per-document delete and minor repairs:
> - `vector_database_fixed.py`
> - `structured_store_fixed.py`
> - `free_rag_system_fixed.py`
>
> **Option A**: rename these to overwrite your originals, or  
> **Option B**: update your imports to use the `_fixed` modules.


## ✅ Requirements

- **Python** 3.9+
- **Ollama** 0.11.4 (daemon reachable at `http://localhost:11434`)
- **No GPU required** (CPU is fine)

If you’re on AlmaLinux and need Python:
```bash
sudo dnf install -y python39 python3-pip
```


## 📦 Python Dependencies

### Minimal (backend + Streamlit UI)
Create `requirements.txt` with:
```txt
chromadb==0.4.22
duckdb==0.9.2
fastapi==0.110.0
uvicorn==0.29.0
pydantic==1.10.13
requests==2.31.0
loguru==0.7.2
streamlit==1.33.0
```

> Note: We’re **not** installing `hnswlib` for now. The vector store will still work (exact cosine search via NumPy).


Install:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## ⚙️ Configuration

Environment variables (optional):
- `OLLAMA_BASE_URL` — default: `http://localhost:11434/api`
- `OLLAMA_EMBED_MODEL` — default: `nomic-embed-text`
- `OLLAMA_CHAT_MODEL` — e.g., `mistral` or `llama3:8b` etc.
- `VECTOR_DB_PATH` — default: `./free_local_vectors`
- `COLLECTION_NAME` — default: `free_documents`

You can hardcode these in your code or pass via your app’s init.


## 🚀 Start Ollama & Pull Models

```bash
# Ensure the daemon is up
ollama --version         # should show 0.11.4
ollama serve &

# Pull models
ollama pull nomic-embed-text      # for embeddings
ollama pull mistral               # or your preferred chat model
# ollama pull llama3:8b  # example alternative
```


## ▶️ Run the Streamlit App

From the project root (where `streamlit_app.py` lives):

```bash
streamlit run streamlit_app.py
```

Then open the URL it prints (usually `http://localhost:8501`).


## 🧪 Using the Python API

Example (headless usage), assuming you renamed the fixed files to replace originals:

```python
from free_rag_system import FreeRAGSystem

rag = FreeRAGSystem(
    db_path="./free_local_vectors",
    collection_name="free_documents",
    ollama_base_url="http://localhost:11434/api",
    ollama_embed_model="nomic-embed-text",
    ollama_chat_model="mistral",
)

# Ingest an unstructured document (returns document_id)
document_id = rag.ingest_unstructured("docs/your-file.pdf")
print("Document ID:", document_id)

# Query
answer, contexts = rag.query("What does the document say about payments?")
print(answer)

# Delete that unstructured document (all its chunks)
ok = rag.delete_unstructured_document(document_id)
print("Vector delete:", ok)

# Ingest a structured file (returns file_id)
file_id = rag.structured_store.ingest_csv("data/sample.csv")
print("File ID:", file_id)

# Delete the structured file (drops its tables + metadata)
ok = rag.delete_structured_file(file_id)
print("Structured delete:", ok)
```


## 🗂️ Where things are stored

**Unstructured (vector) store** — folder (default `./free_local_vectors`):
- `free_documents_embeddings.npy`
- `free_documents_docs.jsonl`
- `free_documents_meta.jsonl`
- `free_documents_ids.json`
- (optional) `free_documents_hnsw.bin` + `free_documents_hnsw_meta.json`

**Structured store** — DuckDB file:
- `structured_store.duckdb`


## 🧹 Deletion Details

- **Per-document (vectors)**: `delete_unstructured_document(document_id)`
  - Removes all chunks for that document from embeddings + docs + meta + ids.
  - Clears HNSW index files if present (they’ll rebuild when large enough).

- **Per-file (structured)**: `delete_structured_file(file_id)`
  - Drops all materialized sheet tables for that file.
  - Deletes rows from `files`, `sheets`, `columns`, `xml_docs`, `xml_nodes`.


## 🔧 Troubleshooting

- **Ollama not reachable**: Make sure `ollama serve` is running and that the VM firewall allows localhost access.
- **Embedding model missing**: Run `ollama pull nomic-embed-text`.
- **Dimension mismatch**: Delete the vector folder if you changed embedding model dims between runs.
- **Old DuckDB**: Ensure DuckDB >= 0.9.0 is installed via `pip install duckdb==0.9.2`.
- **No hnswlib**: That’s fine—search will use exact cosine similarity (slower but correct).


## 🧰 (Optional) Handy `run.sh`

Create a small launcher script:

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1) Start Ollama daemon if not running
if ! pgrep -x "ollama" > /dev/null; then
  echo "Starting Ollama daemon..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
fi

# 2) Launch Streamlit UI
exec streamlit run streamlit_app.py
```

Make it executable:
```bash
chmod +x run.sh
```

Run it:
```bash
./run.sh
```


## 📄 License

Private project (internal). Add your own license if needed.
