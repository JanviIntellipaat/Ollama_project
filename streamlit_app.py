# streamlit_app.py
import os
import tempfile
import streamlit as st

from free_rag_system import FreeRAGSystem

st.set_page_config(page_title="Hybrid RAG (Ollama + DuckDB)", layout="wide")
st.title("📚 Hybrid RAG: Structured (DuckDB) + Unstructured (Local Vectors) with Ollama")

# Sidebar config
st.sidebar.header("⚙️ Settings")
ollama_chat_model = st.sidebar.text_input("Ollama chat model", value=os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1:8b"))
ollama_embed_model = st.sidebar.text_input("Ollama embed model", value=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

if st.sidebar.button("Apply models"):
    os.environ["OLLAMA_CHAT_MODEL"] = ollama_chat_model
    os.environ["OLLAMA_EMBED_MODEL"] = ollama_embed_model
    st.sidebar.success("Model settings applied.")

@st.cache_resource(show_spinner=False)
def _init_rag(chat_model: str, embed_model: str):
    os.environ["OLLAMA_CHAT_MODEL"] = chat_model
    os.environ["OLLAMA_EMBED_MODEL"] = embed_model
    try:
        rag = FreeRAGSystem(
            db_path="./free_local_vectors",
            collection_name="free_documents",
            ollama_chat_model=chat_model
        )
        return rag, None
    except Exception as e:
        return None, str(e)

rag, init_err = _init_rag(ollama_chat_model, ollama_embed_model)
if init_err:
    st.error(f"Failed to initialize RAG system: {init_err}")
    st.stop()

st.subheader("📤 Upload documents (DOCX / CSV / XLSX / XML)")
files = st.file_uploader(
    "Drop files here",
    type=["docx", "csv", "tsv", "xlsx", "xls", "xml"],
    accept_multiple_files=True
)
if files:
    with st.spinner("Processing..."):
        ok, fails = 0, []
        for f in files:
            suffix = "." + f.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                rag.add_document(tmp_path)
                ok += 1
            except Exception as e:
                fails.append((f.name, str(e)))
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        st.success(f"Uploaded {ok} files.")
        if fails:
            st.warning("Some files failed:")
            for n, err in fails:
                st.code(f"{n}: {err}")

st.subheader("❓ Ask a question or generate test cases")
mode = st.radio("Mode", options=["Answer", "Generate Gherkin"], horizontal=True)
prompt = st.text_area(
    "Your question/request",
    height=140,
    placeholder="e.g., Show rows where Field='LEI' and Status='Active' OR Generate Gherkin for Buyr/LEI…"
)

if st.button("Run"):
    if not prompt.strip():
        st.warning("Please enter a question or request.")
    else:
        with st.spinner("Thinking..."):
            res = rag.generate_response(prompt, mode='test_generation' if mode == "Generate Gherkin" else 'general')

        st.markdown("### 🧠 Response")
        st.code(res.get("response", ""), language=None)

        if res.get("context_used"):
            with st.expander("🔎 Sources / Context metadata"):
                srcs = res.get("sources", [])
                if not srcs:
                    st.write("No sources returned.")
                else:
                    for i, s in enumerate(srcs, 1):
                        line = f"**{i}.** File: `{s.get('filename')}`"
                        if s.get("sheet_name"):
                            line += f" | Sheet: `{s.get('sheet_name')}`"
                        if s.get("table_name"):
                            line += f" | Table: `{s.get('table_name')}`"
                        if s.get("section_heading"):
                            line += f" | Section: `{s.get('section_heading')}`"
                        if s.get("from_table") is not None:
                            line += f" | From table: `{s.get('from_table')}`"
                        if s.get("similarity") is not None:
                            line += f" | Sim: `{round(s.get('similarity'),3)}`"
                        if s.get("rerank_score") is not None:
                            line += f" | Rerank: `{round(s.get('rerank_score'),3)}`"
                        st.write(line)

st.subheader("📊 Knowledge Base Stats")
stats = rag.get_knowledge_base_stats()
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Unstructured (Local Vectors)**")
    st.json(stats.get("vector_db", {}))
with col2:
    st.markdown("**Structured (DuckDB)**")
    st.json(stats.get("structured_store", {}))

st.caption("Tip: Start Ollama (`ollama serve`), then pull models: `ollama pull nomic-embed-text` and `ollama pull deepseek-r1:8b`. Optional speed-up: `pip install hnswlib`.")
