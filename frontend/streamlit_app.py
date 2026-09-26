"""Streamlit chat UI. Talks to the FastAPI backend over HTTP — no business
logic lives here, so the frontend can be swapped for Next.js later without
touching the API."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="centered")
st.title("🧠 DocuMind")
st.caption("Ask questions about your ingested document corpus. Answers are grounded with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander(f"Sources ({len(message['citations'])})"):
                for c in message["citations"]:
                    label = f"**{c['document_title']}**"
                    if c.get("section_label"):
                        label += f" — {c['section_label']}"
                    st.markdown(f"{label}  \n_{c['snippet']}..._  \nscore: `{c['score']:.3f}`")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"query": prompt}, timeout=60)
                response.raise_for_status()
                data = response.json()
                st.markdown(data["answer"])
                if not data["is_grounded"]:
                    st.info("No confident match was found in the corpus for this question.")
                if data["citations"]:
                    with st.expander(f"Sources ({len(data['citations'])})"):
                        for c in data["citations"]:
                            label = f"**{c['document_title']}**"
                            if c.get("section_label"):
                                label += f" — {c['section_label']}"
                            st.markdown(f"{label}  \n_{c['snippet']}..._  \nscore: `{c['score']:.3f}`")
                st.caption(
                    f"⏱️ {data['latency_ms']} ms · "
                    f"🪙 {data['prompt_tokens'] + data['completion_tokens']} tokens · "
                    f"💵 ${data['estimated_cost_usd']:.5f}"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "citations": data["citations"],
                    }
                )
            except requests.RequestException as e:
                st.error(f"Could not reach the API: {e}")

with st.sidebar:
    st.header("About")
    st.markdown(
        "DocuMind is a production-grade RAG assistant using hybrid retrieval "
        "(dense + keyword), cross-encoder reranking, and grounded citations."
    )
    st.markdown(f"**API:** `{API_URL}`")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
