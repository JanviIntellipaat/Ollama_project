# structured_store.py (DuckDB-based structured store)
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import duckdb  # pip install duckdb
import pandas as pd
import xml.etree.ElementTree as ET
import difflib

logger = logging.getLogger(__name__)

def _safe_table_name(name: str) -> str:
    import string
    name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    if re.match(r'^\d', name):
        name = '_' + name
    return name[:60]

class StructuredDataStore:
    """
    Stores structured files (CSV/XLSX/XML) in DuckDB with metadata.
    - Auto-detects simple SQL intent from NL and executes safely.
    - Can find/preview sheets when DOCX references Excel files/sheets.
    """

    def __init__(self, db_path: str = "./structured_store.duckdb"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return duckdb.connect(self.db_path)

    def _init_db(self):
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS sheets (
                    sheet_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    file_id BIGINT NOT NULL,
                    sheet_name TEXT NOT NULL,
                    table_name TEXT NOT NULL
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS columns (
                    col_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    sheet_id BIGINT NOT NULL,
                    column_name TEXT NOT NULL,
                    dtype TEXT NOT NULL
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS xml_docs (
                    doc_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    file_id BIGINT NOT NULL,
                    root_tag TEXT NOT NULL
                );
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS xml_nodes (
                    node_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    doc_id BIGINT NOT NULL,
                    path TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    text TEXT,
                    attributes_json TEXT
                );
            """)
            con.commit()

    # ---------- INGEST ----------
    def ingest_csv(self, file_path: str) -> int:
        df = pd.read_csv(file_path, dtype=str).fillna("")
        filename = Path(file_path).name
        with self._conn() as con:
            con.execute("INSERT INTO files(filename, file_type) VALUES (?, ?)", [filename, "csv"])
            file_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

            sheet_name = "Sheet1"
            table_name = _safe_table_name(f"{Path(filename).stem}_{sheet_name}")
            con.register('df_tmp', df)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_tmp")
            con.unregister('df_tmp')

            con.execute("INSERT INTO sheets(file_id, sheet_name, table_name) VALUES (?, ?, ?)",
                        [file_id, sheet_name, table_name])
            sheet_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

            for col in df.columns:
                con.execute("INSERT INTO columns(sheet_id, column_name, dtype) VALUES (?, ?, ?)",
                            [sheet_id, str(col), "TEXT"])
            con.commit()
        logger.info(f"Ingested CSV -> file_id={file_id}, table={table_name}, rows={len(df)}")
        return file_id

    def ingest_excel(self, file_path: str) -> int:
        xls = pd.ExcelFile(file_path)
        filename = Path(file_path).name
        with self._conn() as con:
            con.execute("INSERT INTO files(filename, file_type) VALUES (?, ?)", [filename, "xlsx"])
            file_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

            for sheet in xls.sheet_names:
                df = xls.parse(sheet_name=sheet, dtype=str).fillna("")
                table_name = _safe_table_name(f"{Path(filename).stem}_{sheet}")
                con.register('df_tmp', df)
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_tmp")
                con.unregister('df_tmp')

                con.execute("INSERT INTO sheets(file_id, sheet_name, table_name) VALUES (?, ?, ?)",
                            [file_id, sheet, table_name])
                sheet_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
                for col in df.columns:
                    con.execute("INSERT INTO columns(sheet_id, column_name, dtype) VALUES (?, ?, ?)",
                                [sheet_id, str(col), "TEXT"])
            con.commit()
        logger.info(f"Ingested XLSX -> file_id={file_id}, sheets={len(xls.sheet_names)}")
        return file_id

    def ingest_xml(self, file_path: str) -> int:
        filename = Path(file_path).name
        tree = ET.parse(file_path)
        root = tree.getroot()

        with self._conn() as con:
            con.execute("INSERT INTO files(filename, file_type) VALUES (?, ?)", [filename, "xml"])
            file_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

            con.execute("INSERT INTO xml_docs(file_id, root_tag) VALUES (?, ?)", [file_id, root.tag])
            doc_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Flatten nodes (path simplified)
            for elem in root.iter():
                path = "/" + elem.tag  # simplified path
                text = (elem.text or "").strip()
                attrs = json.dumps(elem.attrib, ensure_ascii=False)
                con.execute("""
                    INSERT INTO xml_nodes(doc_id, path, tag, text, attributes_json)
                    VALUES (?, ?, ?, ?, ?)
                """, [doc_id, path, elem.tag, text, attrs])
            con.commit()

        logger.info(f"Ingested XML -> file_id={file_id}, root={root.tag}")
        return file_id

    # ---------- SIMPLE SEARCH (context previews) ----------
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = f"%{query.strip()}%"
        results: List[Dict[str, Any]] = []
        with self._conn() as con:
            # sheet name hits
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

            # column name hits
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
                    "column": row["column_name"],
                    "preview": f"Column match: {row['column_name']} in {row['sheet_name']} ({row['filename']})"
                })

            # data cell scans (per-table small LIKE)
            tables = con.execute("""
                SELECT s.table_name, s.sheet_name, f.filename FROM sheets s
                JOIN files f ON f.file_id = s.file_id
            """).fetchall()
            per_table = max(1, limit // max(1, len(tables)))
            for tn, sheet, fname in tables:
                try:
                    cols = con.execute(f"PRAGMA table_info('{tn}')").df()["name"].tolist()
                    if not cols:
                        continue
                    # Build OR of ILIKE on all columns (text coercion with ::TEXT)
                    like_parts = " OR ".join([f'CAST("{c}" AS TEXT) ILIKE ?' for c in cols])
                    params = [q for _ in cols]
                    df2 = con.execute(
                        f'SELECT * FROM "{tn}" WHERE {like_parts} LIMIT {per_table};',
                        params
                    ).df()
                    if not df2.empty:
                        preview = df2.head(min(5, len(df2))).to_markdown(index=False)
                        results.append({
                            "type": "data_row",
                            "filename": fname,
                            "sheet_name": sheet,
                            "table_name": tn,
                            "preview": preview
                        })
                except Exception as e:
                    logger.debug(f"LIKE scan failed on {tn}: {e}")

            # XML search
            df = con.execute("""
                SELECT n.path, n.tag, n.text, n.attributes_json, f.filename
                FROM xml_nodes n
                JOIN xml_docs d ON d.doc_id = n.doc_id
                JOIN files f ON f.file_id = d.file_id
                WHERE n.tag ILIKE ? OR n.path ILIKE ? OR n.text ILIKE ? OR n.attributes_json ILIKE ?
                LIMIT ?;
            """, [q, q, q, q, limit]).df()
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

    # ---------- NL → SQL (auto) ----------
    def is_structured_query(self, user_query: str) -> bool:
        q = user_query.lower()
        keywords = ["where", "rows", "row", "table", "column", "sheet", "select", "filter", "list all", "equals", "missing", "is null", "is not null"]
        has_kw = any(k in q for k in keywords)
        has_op = any(op in q for op in ["=", " like ", " ilike ", " in (", " not in ("])
        return has_kw or has_op

    def _list_tables(self, con) -> List[Tuple[str, str, str]]:
        df = con.execute("SELECT s.table_name, s.sheet_name, f.filename FROM sheets s JOIN files f ON f.file_id = s.file_id;").df()
        return list(df.itertuples(index=False, name=None))

    def _list_columns(self, con, table_name: str) -> List[str]:
        df = con.execute(f"PRAGMA table_info('{table_name}')").df()
        return df["name"].tolist() if not df.empty else []

    def _guess_table_and_conditions(self, con, user_query: str) -> Optional[Dict[str, Any]]:
        q = user_query
        conds = []
        for m in re.finditer(r"([A-Za-z0-9_]+)\s*=\s*'([^']+)'", q):
            conds.append((m.group(1), "=", m.group(2)))
        for m in re.finditer(r'([A-Za-z0-9_]+)\s*=\s*"([^"]+)"', q):
            conds.append((m.group(1), "=", m.group(2)))
        for m in re.finditer(r"([A-Za-z0-9_]+)\s+is\s+null", q, re.I):
            conds.append((m.group(1), "IS NULL", None))
        for m in re.finditer(r"([A-Za-z0-9_]+)\s+is\s+not\s+null", q, re.I):
            conds.append((m.group(1), "IS NOT NULL", None))

        tables = self._list_tables(con)
        if not tables:
            return None

        preferred = []
        for tn, sheet, fname in tables:
            if fname and fname.lower() in q.lower():
                preferred.append((tn, sheet, fname))
            elif sheet and sheet.lower() in q.lower():
                preferred.append((tn, sheet, fname))
        candidates = preferred if preferred else tables

        best = None
        best_hits = -1
        for tn, sheet, fname in candidates:
            cols = set(self._list_columns(con, tn))
            hits = sum(1 for c, _, _ in conds if _safe_table_name(c) in cols or c in cols)
            if hits > best_hits:
                best_hits = hits
                best = (tn, sheet, fname, cols)

        if not best:
            return None

        tn, sheet, fname, cols = best
        where_parts = []
        params: List[Any] = []
        for c, op, v in conds:
            col = _safe_table_name(c)
            if col not in cols:
                lc = {cc.lower(): cc for cc in cols}
                col = lc.get(c.lower())
                if not col:
                    continue
            if op == "=":
                where_parts.append(f'"{col}" = ?')
                params.append(v)
            elif op == "IS NULL":
                where_parts.append(f'"{col}" IS NULL')
            elif op == "IS NOT NULL":
                where_parts.append(f'"{col}" IS NOT NULL')

        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        sql = f'SELECT * FROM "{tn}"{where_sql} LIMIT 100;'
        return {"sql": sql, "params": tuple(params), "table_name": tn, "sheet_name": sheet, "filename": fname}

    def try_structured_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        if not self.is_structured_query(user_query):
            return None
        with self._conn() as con:
            guess = self._guess_table_and_conditions(con, user_query)
            if not guess:
                return None
            try:
                df = con.execute(guess["sql"], guess["params"]).df()
                preview = "_(no rows matched)_" if df.empty else df.head(min(20, len(df))).to_markdown(index=False)
                return {
                    "type": "structured_sql",
                    "filename": guess["filename"],
                    "sheet_name": guess["sheet_name"],
                    "table_name": guess["table_name"],
                    "sql": guess["sql"],
                    "preview": preview
                }
            except Exception as e:
                logger.debug(f"SQL exec failed: {e}")
                return None

    # ---------- Sheet resolution for DOCX cross-refs ----------
    def find_sheet(self, filename_hint: str = None, sheet_hint: str = None) -> Optional[Dict[str, str]]:
        with self._conn() as con:
            tables = self._list_tables(con)
            if not tables:
                return None
            scored = []
            for tn, sheet, fname in tables:
                score = 0.0
                if filename_hint:
                    if filename_hint.lower() in fname.lower():
                        score += 1.0
                    else:
                        score += difflib.SequenceMatcher(None, filename_hint.lower(), fname.lower()).ratio() * 0.5
                if sheet_hint:
                    score += difflib.SequenceMatcher(None, (sheet_hint or "").lower(), (sheet or "").lower()).ratio()
                scored.append((score, tn, sheet, fname))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] >= 0.6:
                _, tn, sheet, fname = scored[0]
                return {"table": tn, "sheet": sheet, "file": fname}
            return None

    def preview_sheet(self, table_name: str, limit: int = 20, contains_any: List[str] = None) -> str:
        with self._conn() as con:
            df = con.execute(f'SELECT * FROM "{table_name}" LIMIT 5000;').df()
        if contains_any:
            mask = pd.Series(False, index=df.index)
            for kw in contains_any:
                for col in df.columns:
                    if df[col].dtype == object:
                        mask = mask | df[col].astype(str).str.contains(kw, case=False, na=False)
            if mask.any():
                df = df[mask]
        if df.shape[1] > 16:
            df = df.iloc[:, :16]
        return df.head(limit).to_markdown(index=False)

    # ---------- STATS ----------
    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as con:
            try:
                files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                sheets = con.execute("SELECT COUNT(*) FROM sheets").fetchone()[0]
                columns = con.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
                xml_docs = con.execute("SELECT COUNT(*) FROM xml_docs").fetchone()[0]
                xml_nodes = con.execute("SELECT COUNT(*) FROM xml_nodes").fetchone()[0]
            except Exception:
                files = sheets = columns = xml_docs = xml_nodes = 0
        return {
            "files": files,
            "sheets": sheets,
            "columns": columns,
            "xml_docs": xml_docs,
            "xml_nodes": xml_nodes,
            "db_path": self.db_path
        }

    def list_files(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            df = con.execute("SELECT * FROM files").df()
        return df.to_dict(orient="records")

    def list_sheets(self, file_id: int) -> List[Dict[str, Any]]:
        with self._conn() as con:
            df = con.execute("SELECT * FROM sheets WHERE file_id = ?", [file_id]).df()
        return df.to_dict(orient="records")
