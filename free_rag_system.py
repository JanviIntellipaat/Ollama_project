
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Prefer fixed modules if present
try:
    from vector_database_fixed import FreeVectorDatabase as _VectorDB
except Exception:  # pragma: no cover
    from vector_database import FreeVectorDatabase as _VectorDB

try:
    from structured_store_fixed import StructuredDataStore as _StructuredStore
except Exception:  # pragma: no cover
    from structured_store import StructuredDataStore as _StructuredStore

# Try to import a document processor if present; fall back to a simple one.
try:
    from document_processor import DocumentProcessor  # type: ignore
    _HAS_DOC_PROCESSOR = True
except Exception:
    DocumentProcessor = None  # type: ignore
    _HAS_DOC_PROCESSOR = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _join_api(base_api: str, path: str) -> str:
    base = base_api.rstrip('/')
    p = path.lstrip('/')
    return f"{base}/{p}"


class FreeRAGSystem:
    '''
    Orchestrates:
      - Unstructured vector store (local file-backed) using Ollama embeddings
      - Structured store (DuckDB) for CSV/XLSX/JSON/XML
      - Question answering using an Ollama chat model

    Compatible with the bundled Streamlit app.
    '''

    def __init__(
        self,
        db_path: str = './free_local_vectors',
        collection_name: str = 'free_documents',
        ollama_base_url: str = 'http://127.0.0.1:11435/api',
        ollama_chat_model: str = 'mistral',
        # Optional: allow overriding the embedding model
        ollama_embed_model: Optional[str] = None,
        # Behavior if the embedding dimension in the DB doesn't match the current embedding model:
        #  - 'suffix_collection': use collection_name__<embedmodel> automatically (default, safe)
        #  - 'reset': keep collection name but wipe data to avoid dimension mismatch
        #  - 'error': raise a clear exception
        on_embed_dim_mismatch: str = 'suffix_collection',
        # Structured store path
        structured_db_path: str = './structured_store.duckdb',
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.ollama_chat_model = ollama_chat_model
        self.ollama_embed_model = ollama_embed_model or os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
        self.on_embed_dim_mismatch = on_embed_dim_mismatch

        # Init structured store
        self.structured_store = _StructuredStore(db_path=structured_db_path)

        # Init vector DB (handles Ollama availability internally)
        self.vector_db = _VectorDB(
            db_path=self.db_path,
            collection_name=self.collection_name,
            ollama_base_url=self.ollama_base_url,
            ollama_embed_model=self.ollama_embed_model,
        )

        # If there is existing data, ensure embedding dims match the current embedding model
        try:
            self._ensure_embedding_dim_compatible()
        except Exception as e:
            # Give a very actionable hint
            logger.error(f'Embedding dimension check failed: {e}')
            raise

    # --------- Internal helpers ---------
    def _current_embed_dim(self) -> Optional[int]:
        '''Return dimension of current collection if any data exists.'''
        try:
            if getattr(self.vector_db, '_embeddings', None) is None:
                return None
            arr = self.vector_db._embeddings  # type: ignore[attr-defined]
            if arr is None or getattr(arr, 'shape', None) is None:
                return None
            if arr.shape[0] == 0:
                return None
            return int(arr.shape[1])
        except Exception:
            return None

    def _model_embed_dim(self) -> int:
        '''Probe Ollama once to get the dimension for the current embedding model.'''
        probe = 'dimension check'
        vec = self.vector_db.generate_embeddings([probe])[0]  # uses the configured embed model
        if not isinstance(vec, (list, tuple)) or not vec:
            raise RuntimeError('Embedding model returned empty vector.')
        return len(vec)

    def _ensure_embedding_dim_compatible(self) -> None:
        existing_dim = self._current_embed_dim()
        if existing_dim is None:  # empty DB, nothing to do
            return
        model_dim = self._model_embed_dim()
        if existing_dim == model_dim:
            return  # compatible

        # Dimension mismatch detected
        msg = (
            f'Embedding dimension mismatch: DB has dim {existing_dim}, '
            f'current model \'{self.ollama_embed_model}\' produces dim {model_dim}.'
        )
        if self.on_embed_dim_mismatch == 'error':
            raise RuntimeError(msg + " Set on_embed_dim_mismatch to 'suffix_collection' or 'reset'.")

        if self.on_embed_dim_mismatch == 'reset':
            logger.warning(msg + ' Resetting current collection to avoid errors.')
            self.vector_db.reset_database()
            return

        # Default: suffix collection with the embedding model to keep data separate
        safe_suffix = (
            str(self.ollama_embed_model)
            .replace(':', '_').replace('/', '_').replace(' ', '_')
        )
        new_collection = f"{self.collection_name}__{safe_suffix}"
        logger.warning(msg + f' Switching to a new collection: {new_collection}')
        # Recreate the vector DB pointing to the suffixed collection
        self.collection_name = new_collection
        self.vector_db = _VectorDB(
            db_path=self.db_path,
            collection_name=self.collection_name,
            ollama_base_url=self.ollama_base_url,
            ollama_embed_model=self.ollama_embed_model,
        )

    def _ollama_generate(self, prompt: str, system: Optional[str] = None) -> str:
        '''Call Ollama /generate once, return the full response text.'''
        url = _join_api(self.ollama_base_url, '/generate')
        payload: Dict[str, Any] = {
            'model': self.ollama_chat_model,
            'prompt': prompt if system is None else f'System: {system}\n\n{prompt}',
            'stream': False,
        }
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            # Responses have 'response' for generate;
            # if you switch to /chat, adapt accordingly.
            return data.get('response', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama /generate request failed: {e}")
            return f"[Error contacting model '{self.ollama_chat_model}': {e}]"

    # --------- Public API used by the Streamlit app ---------
    def ingest_unstructured(self, file_path: str) -> str:
        '''
        Process and index an unstructured file. Returns the document_id.
        Tries to use 'document_processor' if available; otherwise apply a simple fallback chunker.
        '''
        if _HAS_DOC_PROCESSOR:
            try:
                dp = DocumentProcessor()  # type: ignore[call-arg]
                doc = dp.process(file_path)  # expected shape: {'chunks': [...], 'metadata': {...}}
                if not isinstance(doc, dict) or 'chunks' not in doc:
                    raise ValueError('DocumentProcessor returned unexpected structure.')
                return self.vector_db.add_document(doc)
            except Exception as e:
                logger.warning(f'DocumentProcessor failed, falling back to simple importer: {e}')

        # Fallback: very simple text importer
        doc = self._fallback_ingest_text(file_path)
        return self.vector_db.add_document(doc)

    def _fallback_ingest_text(self, file_path: str) -> Dict[str, Any]:
        '''Minimal text loader + chunker so ingestion still works without the processor.'''
        p = Path(file_path)
        suffix = p.suffix.lower()
        text = ''
        try:
            if suffix in {'.txt', '.md'}:
                text = p.read_text(encoding='utf-8', errors='ignore')
            else:
                # Last resort for binary formats: try to read as text
                text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            raise RuntimeError(f'Cannot read file {file_path}: {e}')

        def _chunk(s: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            parts: List[str] = []
            start = 0
            while start < len(s):
                end = min(start + max_chars, len(s))
                parts.append(s[start:end])
                if end >= len(s):
                    break
                start = end - overlap
                if start < 0:
                    start = 0
            return parts

        chunks = _chunk(text)
        return {
            'chunks': [
                {
                    'text': c,
                    'from_table': False,
                    'table_headers': '',
                    'section_heading': '',
                    'word_count': len(c.split()),
                    'char_count': len(c),
                }
                for c in chunks
            ],
            'metadata': {
                'filename': p.name,
                'file_type': suffix.lstrip('.'),
                'processed_at': '',
            },
        }

    def query(self, question: str, n_results: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
        '''
        Retrieve top-N chunks and ask the chat model to answer using them.
        Returns (answer, retrieved_contexts)
        '''
        contexts = self.vector_db.similarity_search(question, n_results=n_results) or []
        context_strs: List[str] = []
        for c in contexts:
            md = c.get('metadata', {}) or {}
            src = md.get('filename', 'unknown')
            snippet = c.get('text', '')
            context_strs.append(f'[{src}] {snippet}')

        context_block = '\n\n'.join(context_strs[:n_results])
        prompt = (
            "You are a helpful assistant. Answer the user's question strictly using the provided context. "
            "If the answer cannot be found in the context, say you don't know.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer (be concise, cite sources in brackets like [filename]):"
        )
        answer = self._ollama_generate(prompt)
        return answer, contexts

    # --------- Delete operations (used by Streamlit buttons) ---------
    def delete_unstructured_document(self, document_id: str) -> bool:
        '''
        Delete a document (all its chunks) from the local vector store by document_id.
        Returns True if anything was deleted.
        '''
        try:
            return self.vector_db.delete_document(document_id)
        except Exception as e:
            logger.error(f'Vector delete failed: {e}')
            return False

    def delete_structured_file(self, file_id: int) -> bool:
        '''
        Delete a structured artifact by file_id (drops materialized tables and metadata).
        Returns True if a row was deleted.
        '''
        try:
            return self.structured_store.delete_file(int(file_id))
        except Exception as e:
            logger.error(f'Structured delete failed: {e}')
            return False

    # --------- Optional helpers ---------
    def get_stats(self) -> Dict[str, Any]:
        vstats: Dict[str, Any]
        sstats: Dict[str, Any]
        try:
            vstats = self.vector_db.get_database_stats()
        except Exception as e:
            vstats = {'error': f'vector stats failed: {e}'}
        try:
            sstats = self.structured_store.get_stats()
        except Exception as e:
            sstats = {'error': f'structured stats failed: {e}'}
        return {'vector': vstats, 'structured': sstats}
