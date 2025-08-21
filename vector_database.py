# vector_database.py (SQLite-free local store with optional hnswlib)
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional ANN
try:
    import hnswlib  # optional; used if present and enough data
    HNSW_AVAILABLE = True
except Exception:
    HNSW_AVAILABLE = False

# ---------- Ollama helpers ----------
def _ollama_ok(base_url: str) -> bool:
    try:
        r = requests.get(base_url.replace("/api", ""), timeout=2)
        return r.status_code in (200, 404)
    except Exception:
        return False

def _embed_with_ollama(texts: List[str], model: str, base_url: str) -> List[List[float]]:
    out = []
    for t in texts:
        resp = requests.post(
            f"{base_url}/embeddings",
            json={"model": model, "prompt": t},
            timeout=180
        )
        resp.raise_for_status()
        data = resp.json()
        out.append(data["embedding"])
    return out
# -------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (d,), b: (n, d) -> (n,)
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return bn @ an

class FreeVectorDatabase:
    """
    Tiny local vector store (no SQLite / no Chroma).
    Persists to disk:
      - embeddings.npy  (float32 array, shape (N, D))
      - docs.jsonl      (one document per line)
      - meta.jsonl      (one metadata dict per line)
      - ids.json        (list of ids)
      - (optional) hnsw index: .bin + meta.json
    Embeddings generated via Ollama (default: nomic-embed-text).
    """

    def __init__(
        self,
        db_path: str = "./free_local_vectors",
        collection_name: str = "free_documents",
        ollama_base_url: str = "http://localhost:11434/api",
        ollama_embed_model: str = "nomic-embed-text",
        hnsw_threshold: int = 50000,  # build HNSW when rows exceed this
        hnsw_M: int = 16,
        hnsw_efC: int = 200,
        hnsw_efS: int = 50
    ):
        self.root = Path(db_path)
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url
        self.ollama_embed_model = ollama_embed_model

        self.hnsw_threshold = int(hnsw_threshold)
        self.hnsw_M = int(hnsw_M)
        self.hnsw_efC = int(hnsw_efC)
        self.hnsw_efS = int(hnsw_efS)

        if not _ollama_ok(self.ollama_base_url):
            raise RuntimeError(
                f"Ollama not reachable at {self.ollama_base_url}. "
                f"Start with `ollama serve` and pull: `ollama pull {self.ollama_embed_model}`"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        # files
        self.fp_emb = self.root / f"{collection_name}_embeddings.npy"
        self.fp_docs = self.root / f"{collection_name}_docs.jsonl"
        self.fp_meta = self.root / f"{collection_name}_meta.jsonl"
        self.fp_ids  = self.root / f"{collection_name}_ids.json"

        # hnsw index files
        self.fp_hnsw = self.root / f"{collection_name}_hnsw.bin"
        self.fp_hmeta = self.root / f"{collection_name}_hnsw_meta.json"

        # load existing
        self._embeddings: Optional[np.ndarray] = None
        if self.fp_emb.exists():
            try:
                self._embeddings = np.load(self.fp_emb)
            except Exception as e:
                logger.error(f"Could not load embeddings file: {e}")
                self._embeddings = None
        self._ids: List[str] = []
        if self.fp_ids.exists():
            try:
                self._ids = json.loads(self.fp_ids.read_text())
            except Exception as e:
                logger.error(f"Could not load ids file: {e}")
                self._ids = []

        self._index = None
        self._index_dim = None
        self._index_ready = False
        self._maybe_load_hnsw()

        logger.info(f"Local vector store ready: {self.root} (count={self.count()})")

    # -------- core ----------
    def count(self) -> int:
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    def _append_lines(self, path: Path, lines: List[str]) -> None:
        with path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip("\n") + "\n")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return _embed_with_ollama(texts, self.ollama_embed_model, self.ollama_base_url)

    def add_document(self, document_content: Dict[str, Any], document_id: Optional[str] = None) -> str:
        import uuid
        if document_id is None:
            document_id = str(uuid.uuid4())

        chunks = document_content.get('chunks', [])
        if not chunks:
            raise ValueError("No chunks found in document content")

        docs_lines = []
        meta_lines = []
        chunk_ids = []
        texts = []

        for i, ch in enumerate(chunks):
            texts.append(ch['text'])
            cid = f"{document_id}_chunk_{i}"
            chunk_ids.append(cid)

            md_doc = document_content.get('metadata', {})
            table_headers = ch.get('table_headers', [])
            if isinstance(table_headers, list):
                table_headers = ', '.join(str(x) for x in table_headers) if table_headers else ""
            if table_headers is None:
                table_headers = ""

            external_refs = ch.get('external_refs', "")
            if isinstance(external_refs, (list, dict)):
                external_refs = json.dumps(external_refs, ensure_ascii=False)

            meta = {
                "document_id": str(document_id),
                "chunk_id": int(i),
                "word_count": int(ch.get('word_count', 0) or 0),
                "char_count": int(ch.get('char_count', 0) or 0),
                "filename": str(md_doc.get('filename', '') or ''),
                "file_type": str(md_doc.get('file_type', '') or ''),
                "processed_at": str(md_doc.get('processed_at', '') or ''),
                "from_table": bool(ch.get('from_table', False) or False),
                "table_headers": str(table_headers or ""),
                "section_heading": str(ch.get('section_heading', '') or ''),
                "external_refs": str(external_refs or "")
            }
            docs_lines.append(json.dumps(ch['text'], ensure_ascii=False))
            meta_lines.append(json.dumps(meta, ensure_ascii=False))

        # embed
        try:
            emb = np.asarray(self.generate_embeddings(texts), dtype=np.float32)  # (K, D)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise

        # append to memory + disk
        started_empty = (self._embeddings is None)
        if started_empty:
            self._embeddings = emb
        else:
            if self._embeddings.shape[1] != emb.shape[1]:
                raise ValueError("Embedding dimension mismatch.")
            self._embeddings = np.vstack([self._embeddings, emb])

        # persist
        try:
            np.save(self.fp_emb, self._embeddings)
            self._append_lines(self.fp_docs, docs_lines)
            self._append_lines(self.fp_meta, meta_lines)
            self._ids.extend(chunk_ids)
            self.fp_ids.write_text(json.dumps(self._ids))
        except Exception as e:
            logger.error(f"Persistence error: {e}")
            raise

        # update hnsw if available & large
        try:
            self._maybe_update_hnsw(added=len(texts), dim=self._embeddings.shape[1])
        except Exception as e:
            logger.warning(f"HNSW update failed (non-fatal): {e}")

        logger.info(f"Added document {document_id} with {len(chunks)} chunks (total={self.count()})")
        return document_id

    # -------- HNSW handling ----------
    def _maybe_load_hnsw(self):
        if not HNSW_AVAILABLE:
            return
        if not (self.fp_hnsw.exists() and self.fp_hmeta.exists()):
            return
        try:
            hmeta = json.loads(self.fp_hmeta.read_text())
            dim = int(hmeta.get("dim", 0))
            if dim <= 0:
                return
            index = hnswlib.Index(space="cosine", dim=dim)
            index.load_index(str(self.fp_hnsw))
            efS = int(hmeta.get("efS", 50))
            index.set_ef(efS)
            self._index = index
            self._index_dim = dim
            self._index_ready = True
            logger.info(f"Loaded HNSW index (dim={dim}, efS={efS})")
        except Exception as e:
            logger.warning(f"Could not load HNSW index, will rebuild later: {e}")
            self._index = None
            self._index_ready = False

    def _maybe_update_hnsw(self, added: int, dim: int):
        if not HNSW_AVAILABLE:
            return
        n = self.count()
        if n < self.hnsw_threshold:
            return  # below threshold → no index

        if self._index is None or not self._index_ready or (self._index_dim and self._index_dim != dim):
            # build new index
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=n, ef_construction=self.hnsw_efC, M=self.hnsw_M)
            index.add_items(self._embeddings, np.arange(n))
            index.set_ef(self.hnsw_efS)
            index.save_index(str(self.fp_hnsw))
            self.fp_hmeta.write_text(json.dumps({"dim": dim, "efS": self.hnsw_efS, "M": self.hnsw_M, "efC": self.hnsw_efC}))
            self._index = index
            self._index_dim = dim
            self._index_ready = True
            logger.info(f"Built HNSW index for {n} items (M={self.hnsw_M}, efC={self.hnsw_efC})")
        else:
            # extend existing index
            try:
                current = self._index.get_current_count()
                needed = self.count()
                if needed > self._index.get_max_elements():
                    self._index.resize_index(needed)
                if added > 0:
                    new_range = np.arange(current, current + added)
                    self._index.add_items(self._embeddings[current:current+added], new_range)
                    self._index.save_index(str(self.fp_hnsw))
                    logger.info(f"Extended HNSW index by {added} items (total={needed})")
            except Exception as e:
                logger.warning(f"Extending HNSW index failed; will rebuild later: {e}")
                self._index = None
                self._index_ready = False

    # -------- Query ----------
    def search_similar(self, query: str, n_results: int = 5, filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        total = self.count()
        if total == 0:
            return {"query": query, "total_results": 0, "results": []}

        if n_results <= 0:
            n_results = 1

        # embed query
        try:
            q_emb = np.asarray(self.generate_embeddings([query])[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"Query embedding error: {e}")
            return {"query": query, "total_results": 0, "results": []}

        # HNSW if available & ready
        indices: List[int] = []
        scores: List[float] = []
        if HNSW_AVAILABLE and self._index_ready and total >= self.hnsw_threshold:
            try:
                I, D = self._index.knn_query(q_emb, k=min(n_results, total))
                indices = I[0].tolist()
                scores = [1.0 - float(d) for d in D[0].tolist()]  # cosine distance → similarity
            except Exception as e:
                logger.warning(f"HNSW query failed, falling back to NumPy: {e}")

        # Fallback to exact cosine
        if not indices:
            sims = _cosine_sim(q_emb, self._embeddings)  # (N,)
            topk_idx = np.argpartition(-sims, min(n_results, len(sims)-1))[:n_results]
            topk_sorted = topk_idx[np.argsort(-sims[topk_idx])]
            indices = [int(i) for i in topk_sorted]
            scores = [float(sims[i]) for i in indices]

        # read docs/meta
        results = []
        needed = set(indices)
        docs_by_i: Dict[int, str] = {}
        metas_by_i: Dict[int, Dict[str, Any]] = {}

        try:
            with self.fp_docs.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i in needed:
                        docs_by_i[i] = json.loads(line)
        except Exception as e:
            logger.error(f"Reading docs failed: {e}")
            return {"query": query, "total_results": 0, "results": []}

        try:
            with self.fp_meta.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i in needed:
                        metas_by_i[i] = json.loads(line)
        except Exception as e:
            logger.error(f"Reading meta failed: {e}")
            return {"query": query, "total_results": 0, "results": []}

        for idx, sc in zip(indices, scores):
            meta = metas_by_i.get(idx, {})
            if filter_metadata:
                ok = True
                for k, v in filter_metadata.items():
                    if str(meta.get(k)) != str(v):
                        ok = False
                        break
                if not ok:
                    continue
            results.append({
                "document": docs_by_i.get(idx, ""),
                "metadata": meta,
                "similarity_score": float(sc),
                "chunk_id": self._ids[idx] if idx < len(self._ids) else None
            })

        return {"query": query, "total_results": len(results), "results": results}

    # -------- Stats & maintenance ----------
    def get_database_stats(self) -> Dict[str, Any]:
        total = self.count()
        info = {
            "total_chunks": total,
            "collection_name": self.collection_name,
            "database_path": str(self.root),
            "hnsw_available": HNSW_AVAILABLE,
            "hnsw_enabled": bool(self._index_ready),
            "hnsw_threshold": self.hnsw_threshold
        }
        if self.fp_meta.exists():
            try:
                with self.fp_meta.open("r", encoding="utf-8") as f:
                    for _ in range(10):
                        line = f.readline()
                        if not line:
                            break
                        md = json.loads(line)
                        info.setdefault("file_types", {})
                        ft = md.get("file_type", "unknown")
                        info["file_types"][ft] = info["file_types"].get(ft, 0) + 1
            except Exception:
                pass
        return info

    def delete_document(self, document_id: str) -> bool:
        logger.warning("Selective delete not supported in simple store. Use reset_database().")
        return False

    def reset_database(self) -> bool:
        try:
            for p in [self.fp_emb, self.fp_docs, self.fp_meta, self.fp_ids, self.fp_hnsw, self.fp_hmeta]:
                if p.exists():
                    p.unlink()
            self._embeddings = None
            self._ids = []
            self._index = None
            self._index_ready = False
            logger.info("Local vector store reset.")
            return True
        except Exception as e:
            logger.error(f"Reset error: {e}")
            return False

    def export_metadata(self, output_file: str = None) -> Dict[str, Any]:
        try:
            chunks = []
            if self.fp_meta.exists() and self.fp_docs.exists():
                with self.fp_meta.open("r", encoding="utf-8") as fm, self.fp_docs.open("r", encoding="utf-8") as fd:
                    for i, (ml, dl) in enumerate(zip(fm, fd)):
                        md = json.loads(ml)
                        doc = json.loads(dl)
                        chunks.append({
                            "id": self._ids[i] if i < len(self._ids) else f"row_{i}",
                            "metadata": md,
                            "document_preview": (doc[:200] + "...") if isinstance(doc, str) and len(doc) > 200 else doc
                        })
            export = {
                "export_timestamp": datetime.now().isoformat(),
                "collection_name": self.collection_name,
                "total_chunks": len(chunks),
                "chunks": chunks
            }
            if output_file:
                Path(output_file).write_text(json.dumps(export, indent=2), encoding="utf-8")
                logger.info(f"Exported metadata to {output_file}")
            return export
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {"error": str(e)}
