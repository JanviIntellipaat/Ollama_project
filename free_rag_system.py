# free_rag_system.py
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import requests

from vector_database import FreeVectorDatabase
from structured_store import StructuredDataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _is_structured(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".csv", ".tsv", ".xlsx", ".xls", ".xml"}

# ----------- Ollama chat helper -----------
def ollama_chat(
    messages: List[Dict[str, str]],
    model: str = "deepseek-r1:8b",
    base_url: str = "http://localhost:11434/api",
    temperature: float = 0.1
) -> str:
    resp = requests.post(
        f"{base_url}/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,
                "top_p": 0.9
            }
        },
        timeout=300
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

def _strip_think(content: str) -> str:
    # Remove DeepSeek R1 "thinking" tags if any leak
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
# ------------------------------------------

def _keyword_overlap_score(query: str, text: str) -> float:
    q = re.findall(r"[A-Za-z0-9_]+", query.lower())
    t = text.lower()
    if not q:
        return 0.0
    uniq = set(q)
    overlap = sum(1 for tok in uniq if tok in t)
    for m in re.finditer(r"'([^']+)'", query):
        phrase = m.group(1).lower()
        if phrase and phrase in t:
            overlap += 2
    for m in re.finditer(r'"([^"]+)"', query):
        phrase = m.group(1).lower()
        if phrase and phrase in t:
            overlap += 2
    return overlap / (len(uniq) + 1e-6)

FIELD_WORDS_RE = re.compile(r'(?i)\b(Field|ISIN|LEI|TVTIC|Price|Currency|Error\s*Code|MiFIR|MiFID|Buyer|Seller)\b')

class FreeRAGSystem:
    """
    Hybrid RAG:
      - Structured: DuckDB (CSV/XLSX/XML). Auto-detect SQL intent, execute, inject preview.
      - Unstructured: Local vector store + Ollama embeddings (DOCX). Rerank (semantic + lexical).
      - Cross-ref resolver: links DOCX paragraphs to referenced Excel sheets with previews.
      - Chat: Ollama DeepSeek R1 (configurable).
    """

    def __init__(
        self,
        db_path: str = "./free_local_vectors",
        collection_name: str = "free_documents",
        ollama_base_url: str = "http://localhost:11434/api",
        ollama_chat_model: str = None
    ):
        from document_processor import DocumentProcessor
        self.document_processor = DocumentProcessor()

        self.structured_store = StructuredDataStore(db_path="./structured_store.duckdb")
        self.vector_db = FreeVectorDatabase(
            db_path=db_path,
            collection_name=collection_name,
            ollama_base_url=ollama_base_url,
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        )

        self.ollama_base_url = ollama_base_url
        self.ollama_chat_model = ollama_chat_model or os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1:8b")

        self.system_prompts = {
            'general': (
                "You are a precise assistant for technical/business specifications.\n"
                "- Answer ONLY from the provided context.\n"
                "- Quote field names/error codes verbatim.\n"
                "- Be concise and structured (bullets/tables allowed).\n"
                "- If insufficient context, say so.\n"
                "- Think silently; return only the final answer."
            ),
            'test_generation': (
                "You generate Gherkin feature files from the provided context ONLY.\n"
                "- One scenario per atomic rule/validation.\n"
                "- Use Scenario Outline + Examples when applicable.\n"
                "- Preserve exact field names, values, tags, and error codes.\n"
                "- Include SQL/ref notes as Gherkin comments if present.\n"
                "- Output ONLY valid Gherkin; no explanations.\n"
                "- Think silently; return only the final Gherkin."
            )
        }

        self.conversation_history = []
        logger.info(f"RAG System ready (Ollama chat model: {self.ollama_chat_model}).")

    # ------------ ingestion ------------
    def add_document(self, file_path: str, document_id: str = None) -> str:
        logger.info(f"Ingesting: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if _is_structured(file_path):
            try:
                if ext in (".csv", ".tsv"):
                    fid = self.structured_store.ingest_csv(file_path)
                elif ext in (".xlsx", ".xls"):
                    fid = self.structured_store.ingest_excel(file_path)
                else:  # .xml
                    fid = self.structured_store.ingest_xml(file_path)
                logger.info(f"Structured ingest complete (file_id={fid})")
                return f"struct:{fid}"
            except Exception as e:
                logger.error(f"Structured ingest failed: {e}")
                raise
        else:
            try:
                doc = self.document_processor.process_document(file_path)
                did = self.vector_db.add_document(doc, document_id)
                logger.info(f"Unstructured ingest complete (doc_id={did})")
                return did
            except Exception as e:
                logger.error(f"Unstructured ingest failed: {e}")
                raise

    # ---- hybrid search + rerank + cross-ref ----
    def _collect_keywords(self, text: str) -> List[str]:
        kws = set()
        for m in re.finditer(r'(?i)\b([A-Z][A-Z0-9_]{2,}|[A-Za-z]{3,})\b', text):
            tok = m.group(1)
            if FIELD_WORDS_RE.search(tok) or tok.isupper():
                kws.add(tok.strip(",:.;()[]"))
        return list(kws)[:8]

    def _rerank_unstructured(self, query: str, u_hits: Dict[str, Any], take: int) -> List[Dict[str, Any]]:
        items = []
        for r in u_hits.get("results", []):
            sem = float(r.get("similarity_score", 0.0))
            lex = _keyword_overlap_score(query, r.get("document", ""))
            score = 0.7 * sem + 0.3 * min(1.0, lex)
            r2 = dict(r)
            r2["_rerank_score"] = score
            items.append(r2)
        items.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return items[:take]

    def search_context(self, user_query: str, n_results: int = 8) -> Dict[str, Any]:
        # A) Try NL→SQL for structured intent
        sql_block = self.structured_store.try_structured_query(user_query)

        # B) Structured previews
        s_hits = self.structured_store.search(user_query, limit=max(4, n_results))
        structured_context = ""
        structured_sources = []

        if sql_block:
            prefix = f"\n--- Structured SQL | File: {sql_block.get('filename')} | Sheet: {sql_block.get('sheet_name')} | Table: {sql_block.get('table_name')} ---\n"
            structured_context += prefix + f"```sql\n{sql_block['sql']}\n```\n" + (sql_block.get("preview") or "") + "\n"
            structured_sources.append({
                "filename": sql_block.get("filename"),
                "sheet_name": sql_block.get("sheet_name"),
                "table_name": sql_block.get("table_name"),
                "type": "structured_sql"
            })

        for i, h in enumerate(s_hits):
            prefix = f"\n--- Structured {i+1} | File: {h.get('filename')} | Sheet: {h.get('sheet_name')} | Table: {h.get('table_name')} ---\n"
            structured_context += prefix + (h.get("preview") or "") + "\n"
            structured_sources.append({
                "filename": h.get("filename"),
                "sheet_name": h.get("sheet_name"),
                "table_name": h.get("table_name"),
                "type": h.get("type"),
            })

        # C) Unstructured semantic search + rerank
        rem = max(0, n_results - len(s_hits) - (1 if sql_block else 0))
        u_hits = self.vector_db.search_similar(user_query, n_results=max(6, rem or 6))
        reranked = self._rerank_unstructured(user_query, u_hits, take=max(3, rem or 3))

        unstructured_context = ""
        unstructured_sources = []
        for j, r in enumerate(reranked):
            meta = r["metadata"]
            text = r["document"]
            prefix = f"\n--- Text {j+1}"
            if meta.get('section_heading'):
                prefix += f" (Section: {meta.get('section_heading')})"
            if meta.get('from_table'):
                th = meta.get('table_headers', '')
                prefix += f" [Table: {th}]"
            if meta.get('filename'):
                prefix += f" | File: {meta.get('filename')}"
            prefix += " ---\n"
            block = prefix + text

            # Cross-reference resolver: link DOCX → Excel
            try:
                ext_json = meta.get('external_refs')
                if ext_json:
                    ext_refs = json.loads(ext_json)
                    for ref in ext_refs:
                        if ref.get("type") == "excel":
                            hit = self.structured_store.find_sheet(
                                filename_hint=ref.get("filename_hint"),
                                sheet_hint=ref.get("sheet_hint")
                            )
                            if hit:
                                kw = self._collect_keywords(text)
                                preview = self.structured_store.preview_sheet(hit["table"], limit=15, contains_any=kw)
                                block += (
                                    f"\n\n--- Linked Sheet (auto-resolved) ---\n"
                                    f"File: {hit['file']} | Sheet: {hit.get('sheet') or hit['table']}\n"
                                    f"{preview}\n"
                                )
            except Exception as _:
                logger.debug("Cross-ref resolution failed (non-fatal).", exc_info=False)

            unstructured_context += block + "\n"
            unstructured_sources.append({
                "filename": meta.get("filename", "Unknown"),
                "chunk_id": meta.get("chunk_id", j),
                "section_heading": meta.get("section_heading"),
                "from_table": meta.get("from_table", False),
                "table_headers": meta.get("table_headers", ""),
                "similarity": r.get("similarity_score", 0.0),
                "rerank_score": r.get("_rerank_score", 0.0)
            })

        context_text = structured_context + unstructured_context
        sources = structured_sources + unstructured_sources
        total = len(s_hits) + (1 if sql_block else 0) + len(reranked)
        return {"context": context_text, "sources": sources, "total_results": total}

    def generate_response(self, user_query: str, mode: str = 'general', include_sources: bool = True) -> Dict[str, Any]:
        search_results = self.search_context(user_query, n_results=24 if mode == "test_generation" else 8)
        if search_results['total_results'] == 0:
            return {
                'response': "No relevant information found in the knowledge base.",
                'sources': [],
                'context_used': False
            }

        system_prompt = self.system_prompts.get(mode, self.system_prompts['general'])
        user_prompt = (
            "Answer strictly from the following context.\n\n"
            f"Context:\n{search_results['context']}\n\n"
            f"User Question:\n{user_query}\n\n"
            "Follow the instructions exactly."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if self.conversation_history:
            last = self.conversation_history[-1]
            messages.append({"role": "user", "content": last['user']})
            messages.append({"role": "assistant", "content": last['assistant']})
        messages.append({"role": "user", "content": user_prompt})

        try:
            ai_response = ollama_chat(
                messages,
                model=self.ollama_chat_model,
                base_url=self.ollama_base_url,
                temperature=0.1
            )
            ai_response = _strip_think(ai_response)
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return {'response': f"Error: {e}", 'sources': [], 'context_used': False}

        self.conversation_history.append({
            'user': user_query,
            'assistant': ai_response,
            'timestamp': datetime.now().isoformat(),
            'sources_count': len(search_results['sources'])
        })

        return {
            'response': ai_response,
            'sources': search_results['sources'] if include_sources else [],
            'context_used': True
        }

    def chat(self, user_input: str) -> str:
        return self.generate_response(user_input).get('response', '')

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        stats = self.vector_db.get_database_stats()
        sstats = self.structured_store.get_stats()
        return {"vector_db": stats, "structured_store": sstats}
