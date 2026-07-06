"""Document loaders for PDF and plain-text / Markdown files.

Returns a normalised `Document` (raw text + source metadata). `pypdf` is
imported lazily so text/markdown ingestion works even without it installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS


@dataclass
class Document:
    content: str
    source: str          # filename (basename)
    source_type: str     # 'pdf' | 'markdown' | 'text'
    metadata: dict = field(default_factory=dict)


def _source_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return "pdf"
    if ext in {".md", ".markdown"}:
        return "markdown"
    return "text"


def load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        content=text,
        source=path.name,
        source_type=_source_type(path),
        metadata={"path": str(path), "chars": len(text)},
    )


def load_pdf_file(path: Path) -> Document:
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[page {i + 1}]\n{page_text}")
    text = "\n\n".join(pages)
    return Document(
        content=text,
        source=path.name,
        source_type="pdf",
        metadata={"path": str(path), "pages": len(reader.pages)},
    )


def load_file(path: str | os.PathLike) -> Document:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in PDF_EXTS:
        return load_pdf_file(p)
    if ext in TEXT_EXTS:
        return load_text_file(p)
    raise ValueError(f"Unsupported file type: {p.suffix} ({p})")


def load_path(path: str | os.PathLike) -> list[Document]:
    """Load a single file or every supported file under a directory (recursive)."""
    p = Path(path)
    if p.is_file():
        return [load_file(p)]
    docs: list[Document] = []
    for child in sorted(p.rglob("*")):
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
            docs.append(load_file(child))
    return docs
