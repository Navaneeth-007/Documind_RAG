"""
Chunking strategy: recursive splitting, not fixed-width windows.

Why: fixed-size chunking (e.g. every 500 characters) frequently slices a
sentence or a table row in half, which both hurts embedding quality (the
chunk no longer represents one coherent idea) and hurts citation quality
(you can't point a user to a clean "Section 3.2" reference if the chunk
boundary falls mid-paragraph).

Recursive splitting tries progressively finer separators (section headers ->
paragraphs -> sentences -> words) so most chunks end on a natural boundary,
and only falls back to a hard word-count cut when a single paragraph is
larger than the target chunk size.

A 15% overlap is kept between adjacent chunks so that an answer whose
supporting sentence sits right at a boundary isn't split across two chunks
with neither retrieving cleanly.
"""
import re
from dataclasses import dataclass

from app.ingestion.loader import RawDocument

TARGET_CHUNK_TOKENS = 500
OVERLAP_RATIO = 0.15
# Rough heuristic: ~4 characters per token for English text.
CHARS_PER_TOKEN = 4
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = int(TARGET_CHUNK_CHARS * OVERLAP_RATIO)

SECTION_HEADER_RE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Za-z0-9 ]{2,60}:?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    document_title: str
    source_path: str
    chunk_index: int
    content: str
    section_label: str | None


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on markdown-style or ALL-CAPS/Title-Case headers, keeping the header as a label."""
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    for i, match in enumerate(matches):
        label = match.group().strip("# ").strip().rstrip(":").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((label, body))
    return sections or [(None, text)]


def _split_paragraph_by_size(paragraph: str, max_chars: int) -> list[str]:
    """Fallback: hard-split an oversized paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            current = current[-OVERLAP_CHARS:] + " " + sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_document(doc: RawDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for label, section_text in _split_into_sections(doc.text):
        paragraphs = [p for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        buffer = ""
        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= TARGET_CHUNK_CHARS:
                buffer = candidate
                continue
            if buffer:
                chunks.append(Chunk(doc.title, doc.source_path, idx, buffer, label))
                idx += 1
                buffer = buffer[-OVERLAP_CHARS:]
            if len(para) > TARGET_CHUNK_CHARS:
                for piece in _split_paragraph_by_size(para, TARGET_CHUNK_CHARS):
                    chunks.append(Chunk(doc.title, doc.source_path, idx, piece, label))
                    idx += 1
                buffer = ""
            else:
                buffer = para
        if buffer:
            chunks.append(Chunk(doc.title, doc.source_path, idx, buffer, label))
            idx += 1
    return chunks
