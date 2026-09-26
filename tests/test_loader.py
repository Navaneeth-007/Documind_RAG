from app.ingestion.loader import load_document_from_bytes


def test_load_text_from_bytes():
    raw = b"# Header\nThis is test markdown content."
    doc = load_document_from_bytes("policy.md", raw)
    assert doc is not None
    assert doc.title == "policy"
    assert "This is test markdown content." in doc.text


def test_load_json_from_bytes():
    raw = b'{"policy_name": "remote_work", "stipend": 500}'
    doc = load_document_from_bytes("remote.json", raw)
    assert doc is not None
    assert doc.title == "remote"
    assert "remote_work" in doc.text


def test_load_unsupported_extension_returns_none():
    doc = load_document_from_bytes("image.png", b"\x89PNG\r\n\x1a\n")
    assert doc is None
