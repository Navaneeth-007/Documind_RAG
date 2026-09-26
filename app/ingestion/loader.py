"""
Loads raw documents from a source directory. Kept intentionally simple —
swap in unstructured.io or a cloud document loader if you need OCR /
scanned-PDF support; the interface (`load_documents` returning a list of
RawDocument) is what the rest of the pipeline depends on, so it can be
upgraded independently.
"""
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class RawDocument:
    source_path: str
    title: str
    text: str


def load_documents(source_dir: str) -> list[RawDocument]:
    """Load all supported files from a directory (recursively)."""
    path = Path(source_dir)
    if not path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    documents: list[RawDocument] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text_content = file_path.read_text(encoding="utf-8", errors="ignore")
        if not text_content.strip():
            continue
        documents.append(
            RawDocument(
                source_path=str(file_path),
                title=file_path.stem,
                text=text_content,
            )
        )
    return documents
