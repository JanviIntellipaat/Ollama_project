# structured_store.py — DuckDB portable version (no IDENTITY, no last_insert_rowid)
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import duckdb
import pandas as pd
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def _safe_table_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_",) else "_" for ch in name)

class StructuredDataStore:
    """
    Stores structured artifacts derived from CSV, XLSX, JSON, XML in DuckDB.
    Tables:
      - files(file_id BIGINT PRIMARY KEY, filename TEXT, file_type TEXT)
      - sheets(sheet_id BIGINT PRIMARY KEY, file_id BIGINT, sheet_name TEXT, table_name TEXT)
      - columns(col_id BIGINT PRIMARY KEY, sheet_id BIGINT, column_name TEXT, dtype TEXT)
      - xml_docs(xml_id BIGINT PRIMARY KEY, file_id BIGINT, filename TEXT, root_tag TEXT, raw TEXT)
      - xml_nodes(node_id BIGINT PRIMARY KEY, file_id BIGINT, filename TEXT, path TEXT, tag TEXT, attributes_json TEXT, text TEXT)
    """
    def __init__(self, db_path: str = "./structured_store.duckdb"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return duckdb.connect(self.db_path)

    # ----------- ID helpers (portable) -----------
    def _next_id(self, con: duckdb.DuckDBPyConnection, table: str, id_col: str) -> int:
        try:
            row = con.execute(f"SELECT COALESCE(MAX({id_col}), 0) + 1 FROM {table}").fetchone()
            return int(row[0] or 1)
        except Exception:
            return 1

    def _init_db(self):
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id BIGINT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS sheets (
                    sheet_id BIGINT PRIMARY KEY,
                    file_id BIGINT NOT NULL,
                    sheet_name TEXT NOT NULL,
                    table_name TEXT NOT NULL
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS columns (
                    col_id BIGINT PRIMARY KEY,
                    sheet_id BIGINT NOT NULL,
                    column_name TEXT NOT NULL,
                    dtype TEXT
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS xml_docs (
                    xml_id BIGINT PRIMARY KEY,
                    file_id BIGINT NOT NULL,
                    filename TEXT NOT NULL,
                    root_tag TEXT,
                    raw TEXT
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS xml_nodes (
                    node_id BIGINT PRIMARY KEY,
                    file_id BIGINT NOT NULL,
                    filename TEXT,
                    path TEXT,
                    tag TEXT,
                    attributes_json TEXT,
                    text TEXT
                );
            """)
            con.commit()

    # ---------------- Ingestion ----------------
    def ingest_csv(self, file_path: str) -> int:
        df = pd.read_csv(file_path, dtype=str).fillna("")
        filename = Path(file_path).name
        with self._conn() as con:
            file_id = self._next_id(con, "files", "file_id")
            con.execute("INSERT INTO files(file_id, filename, file_type) VALUES (?, ?, ?)",
                        [file_id, filename, "csv"])

            sheet_name = "Sheet1"
            table_name = _safe_table_name(f"{Path(filename).stem}_{sheet_name}")
            con.register('df_tmp', df)
            con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_tmp')
            con.unregister('df_tmp')

            sheet_id = self._next_id(con, "sheets", "sheet_id")
            con.execute("INSERT INTO sheets(sheet_id, file_id, sheet_name, table_name) VALUES (?, ?, ?, ?)",
                        [sheet_id, file_id, sheet_name, table_name])

            for col in df.columns:
                col_id = self._next_id(con, "columns", "col_id")
                con.execute("INSERT INTO columns(col_id, sheet_id, column_name, dtype) VALUES (?, ?, ?, ?)",
                            [col_id, sheet_id, str(col), "TEXT"])
            con.commit()
        logger.info(f"Ingested CSV -> file_id={file_id}, table={table_name}, rows={len(df)}")
        return file_id

    def ingest_excel(self, file_path: str) -> int:
        xls = pd.ExcelFile(file_path)
        filename = Path(file_path).name
        with self._conn() as con:
            file_id = self._next_id(con, "files", "file_id")
            con.execute("INSERT INTO files(file_id, filename, file_type) VALUES (?, ?, ?)",
                        [file_id, filename, "xlsx"])
            for sheet in xls.sheet_names:
                df = xls.parse(sheet).astype(str).fillna("")
                table_name = _safe_table_name(f"{Path(filename).stem}_{sheet}")
                con.register('df_tmp', df)
                con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_tmp')
                con.unregister('df_tmp')

                sheet_id = self._next_id(con, "sheets", "sheet_id")
                con.execute("INSERT INTO sheets(sheet_id, file_id, sheet_name, table_name) VALUES (?, ?, ?, ?)",
                            [sheet_id, file_id, sheet, table_name])
                for col in df.columns:
                    col_id = self._next_id(con, "columns", "col_id")
                    con.execute("INSERT INTO columns(col_id, sheet_id, column_name, dtype) VALUES (?, ?, ?, ?)",
                                [col_id, sheet_id, str(col), "TEXT"])
            con.commit()
        logger.info(f"Ingested XLSX -> file_id={file_id}, sheets={len(xls.sheet_names)}")
        return file_id

    def ingest_json(self, file_path: str) -> int:
        filename = Path(file_path).name
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data).astype(str).fillna("")
        elif isinstance(data, dict):
            df = pd.json_normalize(data).astype(str).fillna("")
        else:
            df = pd.DataFrame({"value": [str(data)]}).astype(str)

        with self._conn() as con:
            file_id = self._next_id(con, "files", "file_id")
            con.execute("INSERT INTO files(file_id, filename, file_type) VALUES (?, ?, ?)",
                        [file_id, filename, "json"])
            sheet_name = "root"
            table_name = _safe_table_name(f"{Path(filename).stem}_{sheet_name}")
            con.register('df_tmp', df)
            con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_tmp')
            con.unregister('df_tmp')

            sheet_id = self._next_id(con, "sheets", "sheet_id")
            con.execute("INSERT INTO sheets(sheet_id, file_id, sheet_name, table_name) VALUES (?, ?, ?, ?)",
                        [sheet_id, file_id, sheet_name, table_name])
            for col in df.columns:
                col_id = self._next_id(con, "columns", "col_id")
                con.execute("INSERT INTO columns(col_id, sheet_id, column_name, dtype) VALUES (?, ?, ?, ?)",
                            [col_id, sheet_id, str(col), "TEXT"])
            con.commit()
        logger.info(f"Ingested JSON -> file_id={file_id}, table={table_name}, rows={len(df)}")
        return file_id

    def ingest_xml(self, file_path: str) -> int:
        filename = Path(file_path).name
        tree = ET.parse(file_path)
        root = tree.getroot()
        raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")

        with self._conn() as con:
            file_id = self._next_id(con, "files", "file_id")
            con.execute("INSERT INTO files(file_id, filename, file_type) VALUES (?, ?, ?)",
                        [file_id, filename, "xml"])
            xml_id = self._next_id(con, "xml_docs", "xml_id")
            con.execute("INSERT INTO xml_docs(xml_id, file_id, filename, root_tag, raw) VALUES (?, ?, ?, ?, ?)",
                        [xml_id, file_id, filename, root.tag, raw])

            def rec(node, path):
                attrs = json.dumps(node.attrib, ensure_ascii=False)
                text = (node.text or "").strip()
                node_id = self._next_id(con, "xml_nodes", "node_id")
                con.execute("INSERT INTO xml_nodes(node_id, file_id, filename, path, tag, attributes_json, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [node_id, file_id, filename, path, node.tag, attrs, text])
                for i, child in enumerate(list(node)):
                    rec(child, f"{path}/{child.tag}[{i}]")

            rec(root, f"/{root.tag}[0]")
            con.commit()

        logger.info(f"Ingested XML -> file_id={file_id}, nodes stored in xml_nodes")
        return file_id

    # ---------------- Search ----------------
    def simple_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = f"%{query}%"
        results: List[Dict[str, Any]] = []
        with self._conn() as con:
            df = con.execute("""
                SELECT s.table_name, s.sheet_name, f.filename
                FROM sheets s
                JOIN files f ON f.file_id = s.file_id
                WHERE s.sheet_name ILIKE ?
                LIMIT ?;
            """, [q, limit]).df()
            for _, row in df.iterrows():
                results.append({
                    "type": "meta_sheet",
                    "filename": row["filename"],
                    "sheet_name": row["sheet_name"],
                    "table_name": row["table_name"],
                    "preview": f"Sheet match: {row['sheet_name']} in {row['filename']}"
                })

            df = con.execute("""
                SELECT s.table_name, c.column_name, s.sheet_name, f.filename
                FROM columns c
                JOIN sheets s ON s.sheet_id = c.sheet_id
                JOIN files f ON f.file_id = s.file_id
                WHERE c.column_name ILIKE ?
                LIMIT ?;
            """, [q, limit]).df()
            for _, row in df.iterrows():
                results.append({
                    "type": "meta_column",
                    "filename": row["filename"],
                    "sheet_name": row["sheet_name"],
                    "table_name": row["table_name"],
                    "preview": f"Column match: {row['column_name']} in {row['sheet_name']}"
                })

            df = con.execute("""
                SELECT filename, path, tag, attributes_json, text
                FROM xml_nodes
                WHERE text ILIKE ?
                LIMIT ?;
            """, [q, limit]).df()
            for _, row in df.iterrows():
                attrs = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
                attr_str = " ".join([f'{k}="{v}"' for k, v in attrs.items()])
                snippet = f"<{row['tag']} {attr_str}>{(row['text'] or '')[:120]}</{row['tag']}>"
                results.append({
                    "type": "xml_node",
                    "filename": row["filename"],
                    "sheet_name": None,
                    "table_name": None,
                    "preview": f"Path: {row['path']}\nSnippet: {snippet}"
                })
        return results[:limit]

    # ---------------- Admin / Stats ----------------
    def list_files(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            df = con.execute("SELECT * FROM files ORDER BY file_id").df()
        return df.to_dict(orient="records")

    def list_sheets(self, file_id: int) -> List[Dict[str, Any]]:
        with self._conn() as con:
            df = con.execute("SELECT * FROM sheets WHERE file_id = ? ORDER BY sheet_id", [file_id]).df()
        return df.to_dict(orient="records")

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as con:
            files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            sheets = con.execute("SELECT COUNT(*) FROM sheets").fetchone()[0]
            cols = con.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
            xml_docs = con.execute("SELECT COUNT(*) FROM xml_docs").fetchone()[0]
            xml_nodes = con.execute("SELECT COUNT(*) FROM xml_nodes").fetchone()[0]
        return {
            "db_path": self.db_path,
            "files": files,
            "sheets": sheets,
            "columns": cols,
            "xml_docs": xml_docs,
            "xml_nodes": xml_nodes,
        }

    # ---------------- Deletion ----------------
    def delete_file(self, file_id: int) -> bool:
        """Drop tables & metadata for a file_id."""
        with self._conn() as con:
            tables = con.execute("SELECT table_name FROM sheets WHERE file_id = ?", [file_id]).df()["table_name"].tolist()
            for tn in tables:
                try:
                    con.execute(f'DROP TABLE IF EXISTS "{tn}"')
                except Exception as e:
                    logger.warning(f"Failed to drop table {tn}: {e}")

            con.execute("DELETE FROM columns WHERE sheet_id IN (SELECT sheet_id FROM sheets WHERE file_id = ?)", [file_id])
            con.execute("DELETE FROM sheets WHERE file_id = ?", [file_id])
            con.execute("DELETE FROM xml_nodes WHERE file_id = ?", [file_id])
            con.execute("DELETE FROM xml_docs WHERE file_id = ?", [file_id])
            cur = con.execute("DELETE FROM files WHERE file_id = ? RETURNING 1", [file_id])
            deleted = cur.fetchone() is not None
            con.commit()
            return bool(deleted)
