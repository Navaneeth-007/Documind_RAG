"""
Loads raw documents from a source directory or raw uploaded bytes.
Supports Markdown (.md), Plain Text (.txt), PDF (.pdf), JSON (.json), and CSV (.csv).
"""
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("documind.loader")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf"}


@dataclass
class RawDocument:
    source_path: str
    title: str
    text: str


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using pypdf."""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{page_text}")
        return "\n\n".join(pages_text)
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        return ""


def load_document_from_bytes(filename: str, content: bytes) -> RawDocument | None:
    """Parse raw uploaded file bytes into a RawDocument."""
    suffix = Path(filename).suffix.lower()
    title = Path(filename).stem

    if suffix in {".txt", ".md"}:
        text_content = content.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        text_content = _extract_text_from_pdf(content)
    elif suffix == ".json":
        try:
            parsed = json.loads(content.decode("utf-8", errors="ignore"))
            text_content = json.dumps(parsed, indent=2)
        except Exception:
            text_content = content.decode("utf-8", errors="ignore")
    elif suffix == ".csv":
        text_content = content.decode("utf-8", errors="ignore")
    else:
        logger.warning(f"Unsupported file extension: {suffix}")
        return None

    if not text_content.strip():
        return None

    return RawDocument(
        source_path=filename,
        title=title,
        text=text_content,
    )


def load_documents(source_dir: str) -> list[RawDocument]:
    """Load all supported files from a directory (recursively)."""
    path = Path(source_dir)
    if not path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    documents: list[RawDocument] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        try:
            raw_bytes = file_path.read_bytes()
            doc = load_document_from_bytes(file_path.name, raw_bytes)
            if doc:
                doc.source_path = str(file_path)
                documents.append(doc)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")

    return documents
