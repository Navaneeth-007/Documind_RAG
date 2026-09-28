"""
Streamlit Web UI for DocuMind RAG Assistant.
Provides Chat Q&A with live citations, Knowledge Base management,
LLMOps observability metrics, and the Evaluation Benchmark suite.
"""
import json
import os
import sys
import time

# Ensure project root is in sys.path for importing 'eval' and 'app' modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
if not API_URL.startswith("http://") and not API_URL.startswith("https://"):
    if "." not in API_URL and ":" not in API_URL:
        API_URL = f"{API_URL}.onrender.com"
    API_URL = f"https://{API_URL}"

st.set_page_config(
    page_title="DocuMind — Production RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: rgba(240, 242, 246, 0.5);
        border: 1px solid rgba(200, 205, 215, 0.4);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .source-tag {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4F46E5;
        background-color: #EEF2FF;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-right: 6px;
    }
    .grounded-badge {
        color: #059669;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .ungrounded-badge {
        color: #D97706;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def fetch_documents():
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        return r.json() if r.status_code == 200 else {"documents": [], "total_documents": 0, "total_chunks": 0}
    except Exception:
        return {"documents": [], "total_documents": 0, "total_chunks": 0}


def fetch_analytics():
    try:
        r = requests.get(f"{API_URL}/analytics", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# Sidebar
with st.sidebar:
    st.markdown("## 🧠 **DocuMind**")
    st.caption("Production-Grade RAG Architecture")

    health_data = fetch_health()
    if health_data and health_data.get("status") == "ok":
        st.success("🟢 API & Vector DB Connected")
    else:
        st.error("🔴 API or Database Disconnected")

    st.divider()

    st.markdown("### ⚙️ System Configuration")
    if health_data:
        st.write(f"**LLM:** `{health_data.get('llm_provider', 'N/A')}`")
        st.write(f"**Embeddings:** `{health_data.get('embedding_provider', 'N/A')}`")
        st.write(f"**Reranker:** `{health_data.get('reranker_model', 'N/A')}`")

    top_k_input = st.slider("Context Chunks (Top-K)", min_value=1, max_value=10, value=5)
    enable_streaming = st.checkbox("Enable Token Streaming", value=False)

    st.divider()
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Backend URL: `{API_URL}`")

# Tabs
tab_chat, tab_kb, tab_ops, tab_eval = st.tabs(
    ["💬 Assistant & Query", "📚 Knowledge Base", "📊 LLMOps Observability", "🎯 Benchmark & Eval"]
)

# ----------------- TAB 1: Chat -----------------
with tab_chat:
    st.markdown("<h1 class='main-title'>DocuMind Assistant</h1>", unsafe_allow_html=True)
    st.caption("Precision Q&A grounded on your ingested enterprise documents with hybrid retrieval and cross-encoder reranking.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Starter queries
    if not st.session_state.messages:
        st.markdown("##### 💡 Suggested Questions:")
        col1, col2, col3 = st.columns(3)
        sample_q1 = "What is the daily meal limit for international travel?"
        sample_q2 = "What is the policy for reporting suspected security incidents?"
        sample_q3 = "What is the target MTTA for a SEV-1 outage and the escalation path?"

        if col1.button(sample_q1, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sample_q1})
            st.rerun()
        if col2.button(sample_q2, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sample_q2})
            st.rerun()
        if col3.button(sample_q3, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sample_q3})
            st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📑 Sources & Citations ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        title_label = f"**{c['document_title']}**"
                        sec = f" · *{c['section_label']}*" if c.get("section_label") else ""
                        st.markdown(f"{title_label}{sec} `score: {c['score']:.3f}`")
                        st.markdown(f"> {c['snippet']}...")
                        st.divider()

    # Chat Input Handling
    prompt = st.chat_input("Ask a question about your knowledge base...")
    if prompt or (st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and len(st.session_state.messages) % 2 == 1):
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        active_query = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant"):
            with st.spinner("Executing hybrid search & reranking..."):
                try:
                    payload = {
                        "query": active_query,
                        "top_k": top_k_input,
                        "chat_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ],
                    }

                    if enable_streaming:
                        resp = requests.post(f"{API_URL}/query/stream", json=payload, stream=True, timeout=60)
                        resp.raise_for_status()

                        placeholder = st.empty()
                        streamed_text = ""
                        citations_received = []
                        is_grounded = True

                        for line in resp.iter_lines(decode_unicode=True):
                            if line.startswith("data: "):
                                event = json.loads(line[6:])
                                if event["type"] == "content":
                                    streamed_text += event["token"]
                                    placeholder.markdown(streamed_text + "▌")
                                elif event["type"] == "citations":
                                    citations_received = event["citations"]
                                elif event["type"] == "done":
                                    is_grounded = event.get("is_grounded", True)

                        placeholder.markdown(streamed_text)
                        if not is_grounded:
                            st.warning("⚠️ Low retrieval confidence — answer generated from fallback guardrails.")
                        if citations_received:
                            with st.expander(f"📑 Sources & Citations ({len(citations_received)})"):
                                for c in citations_received:
                                    st.markdown(f"**{c['document_title']}** · *{c.get('section_label', '')}* `score: {c['score']:.3f}`")
                                    st.markdown(f"> {c['snippet']}...")

                        st.session_state.messages.append(
                            {"role": "assistant", "content": streamed_text, "citations": citations_received}
                        )

                    else:
                        resp = requests.post(f"{API_URL}/query", json=payload, timeout=180)
                        resp.raise_for_status()
                        data = resp.json()

                        st.markdown(data["answer"])
                        if not data["is_grounded"]:
                            st.warning("⚠️ Low retrieval confidence — answer generated from fallback guardrails.")

                        if data.get("citations"):
                            with st.expander(f"📑 Sources & Citations ({len(data['citations'])})"):
                                for c in data["citations"]:
                                    sec = f" · *{c['section_label']}*" if c.get("section_label") else ""
                                    st.markdown(f"**{c['document_title']}**{sec} `score: {c['score']:.3f}`")
                                    st.markdown(f"> {c['snippet']}...")
                                    st.divider()

                        st.caption(
                            f"⏱️ **Latency:** {data['latency_ms']} ms | "
                            f"🪙 **Tokens:** {data['prompt_tokens'] + data['completion_tokens']} | "
                            f"💵 **Cost:** ${data['estimated_cost_usd']:.6f}"
                        )

                        st.session_state.messages.append(
                            {"role": "assistant", "content": data["answer"], "citations": data.get("citations", [])}
                        )

                except Exception as e:
                    st.error(f"Error calling DocuMind API: {e}")

# ----------------- TAB 2: Knowledge Base -----------------
with tab_kb:
    st.markdown("### 📚 Document Corpus Management")
    st.caption("Upload, inspect, and manage documents indexed in PostgreSQL + pgvector.")

    col_up, col_info = st.columns([1, 1])

    with col_up:
        st.markdown("#### Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files (.md, .txt, .pdf, .json, .csv)",
            type=["md", "txt", "pdf", "json", "csv"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            if st.button("🚀 Ingest Uploaded Files", use_container_width=True):
                with st.spinner("Ingesting and embedding..."):
                    successes = 0
                    for up_file in uploaded_files:
                        files = {"file": (up_file.name, up_file.getvalue(), up_file.type)}
                        r = requests.post(f"{API_URL}/upload", files=files)
                        if r.status_code == 200:
                            successes += 1
                    st.success(f"Successfully indexed {successes} document(s)!")
                    time.sleep(1)
                    st.rerun()

    with col_info:
        st.markdown("#### Direct Text Ingestion")
        with st.form("direct_text_form"):
            doc_title = st.text_input("Document Title", placeholder="e.g., Security FAQ 2026")
            doc_text = st.text_area("Content", placeholder="Paste markdown or raw text here...", height=120)
            submitted = st.form_submit_button("Index Text Document")
            if submitted and doc_title and doc_text:
                r = requests.post(
                    f"{API_URL}/ingest/text",
                    json={"title": doc_title, "content": doc_text, "source_path": "direct_ui_input"},
                )
                if r.status_code == 200:
                    st.success(f"Document '{doc_title}' indexed successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to ingest: {r.text}")

    st.divider()
    st.markdown("#### Indexed Documents")
    doc_data = fetch_documents()
    docs = doc_data.get("documents", [])

    if docs:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total Documents", doc_data.get("total_documents", len(docs)))
        col_m2.metric("Total Chunks in pgvector", doc_data.get("total_chunks", 0))

        df_docs = pd.DataFrame(
            [
                {
                    "ID": d["id"],
                    "Title": d["title"],
                    "Source Path": d["source_path"],
                    "Chunks": d["chunk_count"],
                    "Created At": d["created_at"],
                }
                for d in docs
            ]
        )
        st.dataframe(df_docs, use_container_width=True)

        selected_doc_id = st.selectbox("Select document to remove", options=[d["id"] for d in docs], format_func=lambda x: f"ID {x}: next(d['title'] for d in docs if d['id']==x)")
        if st.button("🗑️ Delete Selected Document", type="secondary"):
            r = requests.delete(f"{API_URL}/documents/{selected_doc_id}")
            if r.status_code == 200:
                st.success(f"Document {selected_doc_id} deleted.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to delete document.")
    else:
        st.info("No documents indexed yet. Upload files above or run the sample ingestion script!")

# ----------------- TAB 3: LLMOps Observability -----------------
with tab_ops:
    st.markdown("### 📊 Production LLMOps & Observability")
    st.caption("Real-time telemetry, latency breakdowns, grounding rates, and token cost tracking.")

    analytics = fetch_analytics()
    if analytics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", analytics.get("total_queries", 0))
        c2.metric("Avg Latency", f"{analytics.get('avg_latency_ms', 0):.0f} ms")
        c3.metric("Total Cost (USD)", f"${analytics.get('total_cost_usd', 0):.5f}")
        c4.metric("Grounding Rate", f"{analytics.get('grounding_rate_pct', 100):.1f}%")

        st.divider()

        recent_logs = analytics.get("recent_logs", [])
        if recent_logs:
            st.markdown("#### Query Activity Logs")
            df_logs = pd.DataFrame(
                [
                    {
                        "ID": entry["id"],
                        "Query": entry["query"],
                        "Answer": entry["answer"][:120] + "..." if entry.get("answer") else "",
                        "Latency (ms)": entry["latency_ms"],
                        "Prompt Tokens": entry["prompt_tokens"],
                        "Completion Tokens": entry["completion_tokens"],
                        "Cost ($)": entry["estimated_cost_usd"],
                        "Time": entry["created_at"],
                    }
                    for entry in recent_logs
                ]
            )
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No query logs recorded yet. Send questions through the chat to populate telemetry.")
    else:
        st.warning("Could not fetch analytics telemetry from API.")

# ----------------- TAB 4: Evaluation Benchmark -----------------
with tab_eval:
    st.markdown("### 🎯 Quantitative Evaluation Harness")
    st.caption("Automated benchmark scoring Faithfulness, Answer Relevance, and Retrieval Precision@k across the golden dataset.")

    col_btn, col_bench = st.columns([1, 2])
    with col_btn:
        st.markdown("#### Run Golden Evaluation")
        st.write("Evaluates the RAG pipeline against `eval/golden_dataset.json`.")
        if st.button("▶️ Execute Benchmark Suite", use_container_width=True):
            with st.spinner("Running evaluation harness..."):
                try:
                    from eval.run_eval import run as run_eval_func

                    results = run_eval_func(api_url=API_URL)
                    st.success("Evaluation completed!")
                except Exception as e:
                    st.error(f"Eval execution failed: {e}")

    # Load existing results if present
    results_file = os.path.join(os.path.dirname(__file__), "..", "eval", "results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                res_data = json.load(f)

            st.divider()
            st.markdown("#### Benchmark Scorecard")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision@5", f"{res_data.get('avg_retrieval_precision', 0):.3f}")
            m2.metric("Answer Relevance", f"{res_data.get('avg_answer_relevance', 0):.3f}")
            m3.metric("Faithfulness", f"{res_data.get('avg_faithfulness', 0):.3f}")
            m4.metric("Avg Latency", f"{res_data.get('avg_latency_ms', 0):.0f} ms")

            st.markdown("#### Detailed Test Breakdown")
            pq = res_data.get("per_question", [])
            if pq:
                df_pq = pd.DataFrame(
                    [
                        {
                            "Question": r["question"],
                            "Precision": round(r["retrieval_precision"], 3),
                            "Relevance": round(r["answer_relevance"], 3),
                            "Faithfulness": round(r["faithfulness"], 3),
                            "Latency (ms)": r["latency_ms"],
                            "Grounded": "✅" if r.get("is_grounded", True) else "⚠️",
                        }
                        for r in pq
                    ]
                )
                st.dataframe(df_pq, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not parse evaluation results: {e}")
