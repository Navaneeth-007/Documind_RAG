"""
CLI entrypoint: chunk + embed + store a directory of documents.

Usage:
    python scripts/ingest_documents.py --source ./data/sample_docs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db import get_connection  # noqa: E402
from app.ingestion.chunker import chunk_document  # noqa: E402
from app.ingestion.embed import embed_texts  # noqa: E402
from app.ingestion.loader import load_documents  # noqa: E402

BATCH_SIZE = 64


def ingest(source_dir: str) -> tuple[int, int]:
    documents = load_documents(source_dir)
    if not documents:
        print(f"No supported documents found in {source_dir} (.txt, .md).")
        return 0, 0

    total_chunks = 0
    with get_connection() as conn:
        for doc in documents:
            doc_id = conn.execute(
                text("INSERT INTO documents (source_path, title) VALUES (:path, :title) RETURNING id"),
                {"path": doc.source_path, "title": doc.title},
            ).scalar_one()

            chunks = chunk_document(doc)
            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[batch_start : batch_start + BATCH_SIZE]
                embeddings = embed_texts([c.content for c in batch])
                for chunk, embedding in zip(batch, embeddings):
                    conn.execute(
                        text(
                            """
                            INSERT INTO chunks
                                (document_id, chunk_index, content, section_label, embedding)
                            VALUES (:doc_id, :idx, :content, :label, :embedding)
                            """
                        ),
                        {
                            "doc_id": doc_id,
                            "idx": chunk.chunk_index,
                            "content": chunk.content,
                            "label": chunk.section_label,
                            "embedding": embedding,
                        },
                    )
            total_chunks += len(chunks)
            print(f"  Ingested '{doc.title}': {len(chunks)} chunks")

    return len(documents), total_chunks


def ingest_api(source_dir: str, api_url: str) -> tuple[int, int]:
    import requests

    documents = load_documents(source_dir)
    if not documents:
        print(f"No supported documents found in {source_dir}.")
        return 0, 0

    total_chunks = 0
    api_url = api_url.rstrip("/")
    if not api_url.startswith("http://") and not api_url.startswith("https://"):
        api_url = f"https://{api_url}"

    print(f"Uploading documents to live API: {api_url} ...")

    for doc in documents:
        resp = requests.post(
            f"{api_url}/ingest/text",
            json={"title": doc.title, "content": doc.text, "source_path": doc.source_path},
            timeout=60,
        )
        if resp.status_code == 200:
            chunks = resp.json().get("chunks_created", 0)
            total_chunks += chunks
            print(f"  ✅ Ingested '{doc.title}': {chunks} chunks")
        else:
            print(f"  ❌ Failed '{doc.title}': {resp.text}")

    return len(documents), total_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into DocuMind.")
    parser.add_argument("--source", required=True, help="Directory of .txt/.md files to ingest")
    parser.add_argument("--api-url", default=None, help="Remote API URL (e.g. https://documind-api-xxxx.onrender.com)")
    args = parser.parse_args()

    if args.api_url:
        n_docs, n_chunks = ingest_api(args.source, args.api_url)
    else:
        print(f"Ingesting documents from {args.source} ...")
        n_docs, n_chunks = ingest(args.source)

    print(f"Done. {n_docs} documents, {n_chunks} chunks ingested.")
