
# vector_database.py (SQLite-free local store with optional hnswlib)
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional ANN (will be ignored if not installed)
try:
    import hnswlib  # type: ignore
    HNSW_AVAILABLE = True
except Exception:
    HNSW_AVAILABLE = False
    hnswlib = None  # type: ignore

# -------- Ollama helpers --------
def _ollama_ok(base_url: str) -> bool:
    try:
        r = requests.get(base_url.replace("/api", "/") + "api/tags", timeout=5)
        return r.status_code == 200
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

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (d,), b: (n, d) -> (n,)
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return bn @ an

class FreeVectorDatabase:
    """
    Tiny local vector store (no SQLite / no Chroma).
    Persists to disk:
      - <collection>_embeddings.npy  (float32 array, shape (N, D))
      - <collection>_docs.jsonl      (one document per line)
      - <collection>_meta.jsonl      (one metadata dict per line)
      - <collection>_ids.json        (list of ids)
      - (optional) HNSW index: <collection>_hnsw.bin + <collection>_hnsw_meta.json
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
        self.fp_hnsw = self.root / f"{collection_name}_hnsw.bin"
        self.fp_hmeta= self.root / f"{collection_name}_hnsw_meta.json"

        # memory
        self._embeddings: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._index = None
        self._index_ready = False
        self._index_dim: Optional[int] = None

        self._load_all()

    # ---------- Load / Save ----------
    def _load_all(self):
        if self.fp_emb.exists():
            try:
                self._embeddings = np.load(self.fp_emb, mmap_mode=None)
            except Exception:
                self._embeddings = None
        if self.fp_ids.exists():
            try:
                self._ids = json.loads(self.fp_ids.read_text(encoding="utf-8"))
            except Exception:
                self._ids = []
        if HNSW_AVAILABLE and self.fp_hnsw.exists() and self.fp_hmeta.exists():
            try:
                meta = json.loads(self.fp_hmeta.read_text(encoding="utf-8"))
                dim = int(meta.get("dim", 0))
                if dim > 0 and self._embeddings is not None and self._embeddings.shape[1] == dim:
                    index = hnswlib.Index(space="cosine", dim=dim)  # type: ignore
                    index.load_index(str(self.fp_hnsw))
                    index.set_ef(int(meta.get("efS", self.hnsw_efS)))
                    self._index = index
                    self._index_dim = dim
                    self._index_ready = True
            except Exception as e:
                logger.warning(f"Failed to load HNSW index: {e}")

    def count(self) -> int:
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    def _append_lines(self, path: Path, lines: List[str]) -> None:
        with path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip("\n") + "\n")

    # ---------- Public API ----------
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return _embed_with_ollama(texts, self.ollama_embed_model, self.ollama_base_url)

    def add_document(self, document_content: Dict[str, Any], document_id: Optional[str] = None) -> str:
        import uuid
        if document_id is None:
            document_id = str(uuid.uuid4())

        chunks = document_content.get('chunks', [])
        if not chunks:
            raise ValueError("No chunks found in document content")

        docs_lines: List[str] = []
        meta_lines: List[str] = []
        chunk_ids: List[str] = []
        texts: List[str] = []

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
        emb = np.asarray(self.generate_embeddings(texts), dtype=np.float32)  # (K, D)

        # append to memory + disk
        started_empty = (self._embeddings is None)
        if started_empty:
            self._embeddings = emb
        else:
            if self._embeddings.shape[1] != emb.shape[1]:
                raise ValueError("Embedding dimension mismatch.")
            self._embeddings = np.vstack([self._embeddings, emb])

        # persist
        np.save(self.fp_emb, self._embeddings)
        self._append_lines(self.fp_docs, docs_lines)
        self._append_lines(self.fp_meta, meta_lines)
        self._ids.extend(chunk_ids)
        self.fp_ids.write_text(json.dumps(self._ids), encoding="utf-8")

        # update hnsw if available & large
        try:
            self._maybe_update_hnsw(added=len(texts), dim=self._embeddings.shape[1])
        except Exception as e:
            logger.warning(f"HNSW update failed (non-fatal): {e}")

        return document_id

    def _maybe_update_hnsw(self, added: int, dim: int):
        if not HNSW_AVAILABLE:
            return
        n = self.count()
        if n < self.hnsw_threshold:
            return  # below threshold → no index

        if self._index is None or not self._index_ready or (self._index_dim and self._index_dim != dim):
            # build new index
            index = hnswlib.Index(space="cosine", dim=dim)  # type: ignore
            index.init_index(max_elements=n, ef_construction=self.hnsw_efC, M=self.hnsw_M)
            index.add_items(self._embeddings, np.arange(n))
            index.set_ef(self.hnsw_efS)
            index.save_index(str(self.fp_hnsw))
            self.fp_hmeta.write_text(json.dumps({"dim": dim, "efS": self.hnsw_efS, "M": self.hnsw_M, "efC": self.hnsw_efC}), encoding="utf-8")
            self._index = index
            self._index_dim = dim
            self._index_ready = True
            logger.info(f"Built HNSW index for {n} items (M={self.hnsw_M}, efC={self.hnsw_efC})")
        else:
            # extend existing index
            current = self._index.get_current_count()
            needed = self.count()
            if needed > self._index.get_max_elements():
                self._index.resize_index(needed)
            if added > 0:
                new_range = np.arange(current, current + added)
                self._index.add_items(self._embeddings[current:current+added], new_range)
                self._index.save_index(str(self.fp_hnsw))

    def similarity_search(self, query: str, n_results: int = 8) -> List[Dict[str, Any]]:
        if self._embeddings is None or self.count() == 0:
            return []
        # embed query
        q_vec = self.generate_embeddings([query])[0]
        q_emb = np.asarray(q_vec, dtype=np.float32)

        total = self.count()
        indices: List[int] = []
        scores: List[float] = []

        # try HNSW
        if HNSW_AVAILABLE and self._index_ready and total >= self.hnsw_threshold:
            try:
                I, D = self._index.knn_query(q_emb, k=min(n_results, total))
                indices = I[0].tolist()
                scores = [1.0 - float(d) for d in D[0].tolist()]  # cosine distance → similarity
            except Exception as e:
                logger.warning(f"HNSW query failed, falling back to NumPy: {e}")

        # Fallback exact cosine
        if not indices:
            sims = _cosine_sim(q_emb, self._embeddings)  # (N,)
            topk_idx = np.argpartition(-sims, min(n_results, len(sims)-1))[:n_results]
            topk_sorted = topk_idx[np.argsort(-sims[topk_idx])]
            indices = [int(i) for i in topk_sorted]
            scores = [float(sims[i]) for i in indices]

        # read docs/meta
        results: List[Dict[str, Any]] = []
        needed = set(indices)
        docs_by_i: Dict[int, str] = {}
        metas_by_i: Dict[int, Dict[str, Any]] = {}

        with self.fp_docs.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in needed:
                    docs_by_i[i] = json.loads(line)
        with self.fp_meta.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in needed:
                    metas_by_i[i] = json.loads(line)

        for rank, idx in enumerate(indices):
            doc = docs_by_i.get(idx, "")
            md = metas_by_i.get(idx, {})
            results.append({
                "score": float(scores[rank]) if rank < len(scores) else None,
                "text": doc,
                "metadata": md
            })
        return results

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
        # sample file types from meta
        file_types: Dict[str, int] = {}
        if self.fp_meta.exists():
            try:
                with self.fp_meta.open("r", encoding="utf-8") as f:
                    for _ in range(1000):  # scan up to first 1000 lines for stats
                        line = f.readline()
                        if not line:
                            break
                        md = json.loads(line)
                        ft = md.get("file_type", "unknown")
                        file_types[ft] = file_types.get(ft, 0) + 1
            except Exception:
                pass
        if file_types:
            info["file_types"] = file_types
        return info

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks (rows) associated with a given document_id.
        Returns True if anything was deleted, False otherwise.
        """
        if not (self.fp_docs.exists() and self.fp_meta.exists() and self.fp_emb.exists()):
            return False
        # Identify indices to keep / drop
        keep_indices: List[int] = []
        drop_indices: List[int] = []
        metas: List[Dict[str, Any]] = []
        with self.fp_meta.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                md = json.loads(line)
                metas.append(md)
                if str(md.get("document_id")) == str(document_id):
                    drop_indices.append(i)
                else:
                    keep_indices.append(i)
        if not drop_indices:
            return False

        # Rebuild embeddings & ids
        emb = np.load(self.fp_emb)
        keep_mask = np.ones(emb.shape[0], dtype=bool)
        keep_mask[drop_indices] = False
        new_emb = emb[keep_mask]
        np.save(self.fp_emb, new_emb)
        self._embeddings = new_emb

        # Rebuild docs.jsonl and meta.jsonl
        new_docs_lines: List[str] = []
        new_meta_lines: List[str] = []
        with self.fp_docs.open("r", encoding="utf-8") as f_docs, self.fp_meta.open("r", encoding="utf-8") as f_meta:
            for i, (dline, mline) in enumerate(zip(f_docs, f_meta)):
                if i in keep_indices:
                    new_docs_lines.append(dline.rstrip("\n"))
                    new_meta_lines.append(mline.rstrip("\n"))
        self.fp_docs.write_text("\n".join(new_docs_lines) + ("\n" if new_docs_lines else ""), encoding="utf-8")
        self.fp_meta.write_text("\n".join(new_meta_lines) + ("\n" if new_meta_lines else ""), encoding="utf-8")

        # Rebuild ids
        new_ids: List[str] = []
        for i, old_id in enumerate(self._ids):
            # old ids were document_id_chunk_k; recompute only for kept rows
            # We cannot know positions here reliably, regenerate sequentially:
            new_ids.append(old_id)  # keep original ids for stability
        # But also drop ids for removed indices:
        new_ids = [id_ for i, id_ in enumerate(self._ids) if i in keep_indices]
        self._ids = new_ids
        self.fp_ids.write_text(json.dumps(self._ids), encoding="utf-8")

        # Remove/refresh HNSW
        if HNSW_AVAILABLE and self.fp_hnsw.exists():
            try:
                self.fp_hnsw.unlink(missing_ok=True)
                self.fp_hmeta.unlink(missing_ok=True)
            except Exception:
                pass
            self._index = None
            self._index_ready = False
            self._index_dim = None
            # Rebuild index only if above threshold
            if self.count() >= self.hnsw_threshold:
                self._maybe_update_hnsw(added=0, dim=self._embeddings.shape[1])

        logger.info(f"Deleted document_id={document_id} with {len(drop_indices)} chunks.")
        return True

    def reset_database(self) -> None:
        """Delete all data in this collection."""
        for fp in [self.fp_emb, self.fp_docs, self.fp_meta, self.fp_ids, self.fp_hnsw, self.fp_hmeta]:
            try:
                fp.unlink(missing_ok=True)
            except Exception:
                pass
        self._embeddings = None
        self._ids = []
        self._index = None
        self._index_ready = False
        self._index_dim = None
        logger.warning("Vector database reset completed.")

    # ---------- Export ----------
    def export_metadata(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        chunks: List[Dict[str, Any]] = []
        try:
            with self.fp_docs.open("r", encoding="utf-8") as f_docs, self.fp_meta.open("r", encoding="utf-8") as f_meta:
                for dline, mline in zip(f_docs, f_meta):
                    doc = json.loads(dline)
                    md = json.loads(mline)
                    chunks.append({
                        "text": doc,
                        "metadata": md,
                        "document_preview": (doc[:200] + "...") if isinstance(doc, str) and len(doc) > 200 else doc
                    })
            export = {
                "collection_name": self.collection_name,
                "database_path": str(self.root),
                "total_chunks": len(chunks),
                "chunks": chunks
            }
            if output_file:
                Path(output_file).write_text(json.dumps(export, indent=2), encoding="utf-8")
                logger.info(f"Exported metadata to {output_file}")
            return export
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
