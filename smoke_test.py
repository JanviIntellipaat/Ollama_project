#!/usr/bin/env python3
import os, json, time, sys
from pathlib import Path

# ---- Config (override via env if needed) ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435/api")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./free_local_vectors")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "free_documents")
ON_EMBED_DIM_MISMATCH = os.getenv("ON_EMBED_DIM_MISMATCH", "suffix_collection")

def log(msg):
    print(f"[SMOKE] {msg}", flush=True)

def check_ollama():
    import requests
    url = OLLAMA_BASE_URL.rstrip("/") + "/tags"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        names = [m.get("name") for m in data.get("models", [])]
        log(f"Ollama reachable at {url}. Models: {names}")
        if not names:
            log("WARNING: No models installed. run.sh should pull 'nomic-embed-text' and 'mistral'.")
        return True
    except Exception as e:
        log(f"ERROR: Ollama API not reachable at {url}: {e}")
        return False

def main():
    ok = check_ollama()
    if not ok:
        log("Aborting smoke test (start ./run.sh first).")
        sys.exit(2)

    # Prefer fixed modules if present
    try:
        from free_rag_system import FreeRAGSystem
    except Exception as e:
        log(f"ERROR: Cannot import FreeRAGSystem: {e}")
        sys.exit(3)

    # Create tiny sample files
    Path("uploads").mkdir(exist_ok=True)
    txt = Path("uploads/sample.txt")
    txt.write_text("Payments are processed on the 15th of each month.\nThis is a test document.", encoding="utf-8")
    csv = Path("uploads/sample.csv")
    csv.write_text("name,amount\nAlice,100\nBob,200\n", encoding="utf-8")

    # Init system
    rag = FreeRAGSystem(
        db_path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_chat_model=OLLAMA_CHAT_MODEL,
        ollama_embed_model=OLLAMA_EMBED_MODEL,
        on_embed_dim_mismatch=ON_EMBED_DIM_MISMATCH,
        structured_db_path="./structured_store.duckdb",
    )

    # Ingest unstructured
    log("Ingesting unstructured: uploads/sample.txt")
    try:
        doc_id = rag.ingest_unstructured(str(txt))
        log(f"OK: document_id={doc_id}")
    except Exception as e:
        log(f"ERROR ingesting unstructured: {e}")
        sys.exit(4)

    # Ingest structured CSV
    log("Ingesting structured CSV: uploads/sample.csv")
    try:
        file_id = rag.structured_store.ingest_csv(str(csv))
        log(f"OK: file_id={file_id}")
    except Exception as e:
        log(f"ERROR ingesting CSV: {e}")
        sys.exit(5)

    # Query
    q = "When are payments processed?"
    log(f"Querying: {q}")
    try:
        answer, ctxs = rag.query(q, n_results=5)
        log(f"Answer (first 240 chars): {answer[:240].replace(chr(10),' ')}")
        log(f"Retrieved {len(ctxs)} contexts")
    except Exception as e:
        log(f"ERROR during query: {e}")
        sys.exit(6)

    # Stats
    try:
        stats = rag.get_stats()
        log("Stats: " + json.dumps(stats, indent=2)[:400].replace("\n"," "))
    except Exception as e:
        log(f"ERROR reading stats: {e}")

    # Delete
    log("Deleting the test resources...")
    try:
        vd = rag.delete_unstructured_document(doc_id)
        sd = rag.delete_structured_file(int(file_id))
        log(f"Delete vector={vd}, structured={sd}")
    except Exception as e:
        log(f"ERROR during delete: {e}")

    log("Smoke test completed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
