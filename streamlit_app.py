
import os
import time
import streamlit as st

# Prefer fixed modules if present
try:
    from free_rag_system_fixed import FreeRAGSystem
except Exception:
    from free_rag_system import FreeRAGSystem

# ---------------- Config ----------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./free_local_vectors")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "free_documents")

# Create a singleton RAG system
@st.cache_resource(show_spinner=False)
def get_rag():
    return FreeRAGSystem(
        db_path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_embed_model=OLLAMA_EMBED_MODEL,
        ollama_chat_model=OLLAMA_CHAT_MODEL,
    )

rag = get_rag()

st.set_page_config(page_title="Ollama_AI_Janvi — RAG", layout="wide")
st.title("🧠 Ollama_AI_Janvi — RAG Console")

tabs = st.tabs(["🔎 Query", "📥 Ingest", "🗑️ Manage / Delete", "📊 Stats"])

# ---------------- Tab: Query ----------------
with tabs[0]:
    st.subheader("Ask your corpus")
    q = st.text_input("Your question")
    k = st.slider("Context chunks", min_value=1, max_value=12, value=6)
    if st.button("Run", type="primary"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, ctxs = rag.query(q, n_results=k)
                    st.markdown("### Answer")
                    st.write(answer)
                    with st.expander("Show retrieved context"):
                        for i, c in enumerate(ctxs):
                            st.markdown(f"**{i+1}.** {c.get('metadata', {}).get('filename','(unknown)')} — section: {c.get('metadata', {}).get('section_heading','')}")
                            st.caption(c.get("text","")[:800])
                except Exception as e:
                    st.error(f"Query failed: {e}")

# ---------------- Tab: Ingest ----------------
with tabs[1]:
    st.subheader("Ingest Files")
    up = st.file_uploader("Upload files (PDF/DOCX/MD/TXT/CSV/XLSX/JSON/XML)", accept_multiple_files=True)
    if up:
        for f in up:
            suffix = (f.name.split(".")[-1] or "").lower()
            tmp_path = Path("uploads")
            tmp_path.mkdir(exist_ok=True)
            save_to = tmp_path / f.name
            with open(save_to, "wb") as out:
                out.write(f.getbuffer())
            try:
                if suffix in {"pdf","docx","md","txt"}:
                    with st.spinner(f"Ingesting unstructured: {f.name}"):
                        did = rag.ingest_unstructured(str(save_to))
                        st.success(f"Ingested (unstructured). document_id={did}")
                elif suffix in {"csv"}:
                    with st.spinner(f"Ingesting CSV: {f.name}"):
                        fid = rag.structured_store.ingest_csv(str(save_to))
                        st.success(f"Ingested CSV. file_id={fid}")
                elif suffix in {"xlsx","xls"}:
                    with st.spinner(f"Ingesting Excel: {f.name}"):
                        fid = rag.structured_store.ingest_excel(str(save_to))
                        st.success(f"Ingested Excel. file_id={fid}")
                elif suffix in {"json"}:
                    with st.spinner(f"Ingesting JSON: {f.name}"):
                        fid = rag.structured_store.ingest_json(str(save_to))
                        st.success(f"Ingested JSON. file_id={fid}")
                elif suffix in {"xml"}:
                    with st.spinner(f"Ingesting XML: {f.name}"):
                        fid = rag.structured_store.ingest_xml(str(save_to))
                        st.success(f"Ingested XML. file_id={fid}")
                else:
                    st.warning(f"Unsupported file type for {f.name}. Skipped.")
            except Exception as e:
                st.error(f"Failed to ingest {f.name}: {e}")

# ---------------- Tab: Manage / Delete ----------------
with tabs[2]:
    col1, col2 = st.columns(2)

    # --- Unstructured (vector) deletion ---
    with col1:
        st.markdown("### Unstructured — Delete by document_id")
        doc_id = st.text_input("document_id to delete", key="doc_id_input")
        if st.button("Delete Document", key="delete_doc_btn"):
            if not doc_id.strip():
                st.warning("Enter a document_id first.")
            else:
                with st.spinner(f"Deleting document_id={doc_id}..."):
                    try:
                        ok = rag.delete_unstructured_document(doc_id)
                        if ok:
                            st.success(f"Deleted document {doc_id} from vector store.")
                        else:
                            st.info(f"No chunks found for document_id={doc_id}.")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

    # --- Structured (DuckDB) deletion ---
    with col2:
        st.markdown("### Structured — Delete by file_id")
        # List files with delete buttons
        try:
            files = rag.structured_store.list_files()
        except Exception as e:
            files = []
            st.error(f"Could not list files: {e}")

        if files:
            for rec in files:
                with st.container(border=True):
                    st.write(f"**file_id:** {rec.get('file_id')}  |  **filename:** {rec.get('filename')}  |  **type:** {rec.get('file_type')}")
                    bcol1, bcol2 = st.columns([1,1])
                    with bcol1:
                        if st.button(f"Delete file_id {rec.get('file_id')}", key=f"del_{rec.get('file_id')}"):
                            with st.spinner(f"Dropping tables and metadata for file_id={rec.get('file_id')}..."):
                                try:
                                    ok = rag.delete_structured_file(int(rec.get('file_id')))
                                    if ok:
                                        st.success(f"Deleted file_id={rec.get('file_id')}")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.info("Nothing deleted (maybe it was already removed).")
                                except Exception as e:
                                    st.error(f"Delete failed: {e}")
                    with bcol2:
                        if st.button(f"View sheets for {rec.get('file_id')}", key=f"view_{rec.get('file_id')}"):
                            try:
                                sheets = rag.structured_store.list_sheets(int(rec.get('file_id')))
                                if sheets:
                                    st.write(sheets)
                                else:
                                    st.info("No sheets found.")
                            except Exception as e:
                                st.error(f"List sheets failed: {e}")
        else:
            st.info("No structured files ingested yet.")

# ---------------- Tab: Stats ----------------
with tabs[3]:
    st.subheader("Database Stats")
    try:
        vstats = rag.vector_db.get_database_stats()
        st.markdown("**Vector store**")
        st.json(vstats)
    except Exception as e:
        st.error(f"Vector stats error: {e}")

    try:
        sstats = rag.structured_store.get_stats()
        st.markdown("**Structured store (DuckDB)**")
        st.json(sstats)
    except Exception as e:
        st.error(f"Structured stats error: {e}")
