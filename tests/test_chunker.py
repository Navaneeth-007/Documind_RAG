from app.ingestion.chunker import chunk_document
from app.ingestion.loader import RawDocument


def test_short_document_produces_single_chunk():
    doc = RawDocument(source_path="x.txt", title="short", text="This is a short paragraph.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "This is a short paragraph."


def test_section_headers_are_preserved_as_labels():
    text = "Introduction:\nThis is the intro paragraph with enough content to stand alone.\n\nMethods:\nThis is the methods paragraph with enough content to stand alone."
    doc = RawDocument(source_path="x.txt", title="sectioned", text=text)
    chunks = chunk_document(doc)
    labels = {c.section_label for c in chunks}
    assert "Introduction" in labels
    assert "Methods" in labels


def test_long_paragraph_is_split_into_multiple_chunks():
    long_paragraph = " ".join(["This is sentence number {}.".format(i) for i in range(400)])
    doc = RawDocument(source_path="x.txt", title="long", text=long_paragraph)
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 2500  # target chars + generous margin for overlap


def test_chunk_indices_are_sequential():
    text = "Para one is here with some words.\n\nPara two is here with some words.\n\nPara three is here."
    doc = RawDocument(source_path="x.txt", title="seq", text=text)
    chunks = chunk_document(doc)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
