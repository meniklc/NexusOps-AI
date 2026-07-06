"""Offline tests for the recursive character splitter."""
from __future__ import annotations

from app.ingestion.splitter import RecursiveCharacterTextSplitter, split_text


def test_short_text_single_chunk():
    chunks = split_text("hello world", chunk_size=100, chunk_overlap=10)
    assert chunks == ["hello world"]


def test_chunks_respect_size_bound():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    # Allow modest slack for separators, but nothing pathological.
    assert all(len(c) <= 300 * 1.2 for c in chunks)


def test_overlap_preserves_context():
    # Distinct sentence tokens let us detect overlap between neighbours.
    sentences = [f"S{i} is a unique marker sentence number {i}. " for i in range(30)]
    text = "".join(sentences)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=60)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    # Consecutive chunks should share some trailing/leading text.
    overlaps = 0
    for a, b in zip(chunks, chunks[1:]):
        tail = a[-40:]
        if any(tok and tok in b for tok in tail.split()):
            overlaps += 1
    assert overlaps >= 1


def test_no_empty_chunks():
    text = "a" * 5000
    chunks = split_text(text, chunk_size=256, chunk_overlap=32)
    assert all(c.strip() for c in chunks)
    assert len(chunks) >= 5


def test_rejects_bad_overlap():
    try:
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= size")
