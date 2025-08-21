# document_processor.py
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from docx import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Reference extractors ---
EXCEL_REF_RE = re.compile(
    r"""(?ix)
    (?:see|refer\s+to|as\s+per|documented\s+in)\s+
    (?:
        (?:sheet|table)\s+["']?([A-Za-z0-9 _\-\.\(\)]+)["']?\s+(?:of|in)\s+
    )?
    ([A-Za-z0-9 _\-\.\(\)]+\.xlsx)
    """.strip()
)
TABLE_HINT_RE = re.compile(r'(?i)\b(Table\s*\d+|Mapping(?:\s*Sheet)?|Error\s*Codes?)\b')
FIELD_TAG_RE = re.compile(r'(?i)\bField\s*(\d+)\s*[:\-–]\s*([A-Za-z0-9 _/\-\(\)]+)')

def _extract_external_refs_from_text(text: str):
    refs = []
    for m in EXCEL_REF_RE.finditer(text):
        sheet_or_table = (m.group(1) or "").strip()
        filename = m.group(2).strip()
        refs.append({
            "type": "excel",
            "filename_hint": filename,
            "sheet_hint": sheet_or_table or None
        })
    for m in TABLE_HINT_RE.finditer(text):
        refs.append({"type": "hint", "table_hint": m.group(0)})
    for m in FIELD_TAG_RE.finditer(text):
        refs.append({"type": "field_ref", "field_no": m.group(1), "field_name": m.group(2).strip()})
    # dedupe
    dedup = []
    seen = set()
    for r in refs:
        k = json.dumps(r, sort_keys=True)
        if k not in seen:
            seen.add(k); dedup.append(r)
    return dedup
# --------------------------------

def _chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        split = text.rfind(".", start, end)
        if split == -1 or split <= start + 200:
            split = end
        chunks.append(text[start:split].strip())
        start = split
    return [c for c in chunks if c]

class DocumentProcessor:
    """
    Process unstructured documents (.docx) into chunks with metadata.
    """

    def process_document(self, file_path: str) -> Dict[str, Any]:
        ext = Path(file_path).suffix.lower()
        if ext != ".docx":
            raise ValueError(f"Unsupported for DocumentProcessor: {ext} (structured files handled elsewhere)")
        return self._process_docx(file_path)

    def _process_docx(self, file_path: str) -> Dict[str, Any]:
        doc = Document(file_path)
        filename = Path(file_path).name
        processed_at = datetime.now().isoformat()

        chunks: List[Dict[str, Any]] = []
        tables_meta: List[Dict[str, Any]] = []

        current_heading = None
        para_buffer: List[str] = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name or "").lower()
            if "heading" in style:
                if para_buffer:
                    full = "\n".join(para_buffer).strip()
                    for ch in _chunk_text(full):
                        refs = _extract_external_refs_from_text(ch)
                        chunks.append({
                            "text": ch,
                            "word_count": len(ch.split()),
                            "char_count": len(ch),
                            "section_heading": current_heading or "",
                            "from_table": False,
                            "table_headers": [],
                            "external_refs": json.dumps(refs, ensure_ascii=False)
                        })
                    para_buffer = []
                current_heading = p.text.strip()
            else:
                para_buffer.append(text)

        if para_buffer:
            full = "\n".join(para_buffer).strip()
            for ch in _chunk_text(full):
                refs = _extract_external_refs_from_text(ch)
                chunks.append({
                    "text": ch,
                    "word_count": len(ch.split()),
                    "char_count": len(ch),
                    "section_heading": current_heading or "",
                    "from_table": False,
                    "table_headers": [],
                    "external_refs": json.dumps(refs, ensure_ascii=False)
                })

        # extract tables (as markdown preview)
        for t in doc.tables:
            rows = []
            for r in t.rows:
                rows.append([cell.text.strip() for cell in r.cells])
            headers = rows[0] if rows else []
            md_lines = []
            if headers:
                md_lines.append("| " + " | ".join(h or "" for h in headers) + " |")
                md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
                for row in rows[1:6]:
                    md_lines.append("| " + " | ".join(row) + " |")
            table_md = "\n".join(md_lines) if md_lines else ""
            if table_md:
                refs = _extract_external_refs_from_text(table_md)
                chunks.append({
                    "text": f"Table:\n{table_md}",
                    "word_count": len(table_md.split()),
                    "char_count": len(table_md),
                    "section_heading": current_heading or "Table",
                    "from_table": True,
                    "table_headers": headers,
                    "external_refs": json.dumps(refs, ensure_ascii=False)
                })
                tables_meta.append({"headers": headers, "rows_preview": rows[1:6]})

        return {
            "chunks": chunks,
            "tables": tables_meta,
            "metadata": {
                "filename": filename,
                "file_type": "docx",
                "processed_at": processed_at
            }
        }
