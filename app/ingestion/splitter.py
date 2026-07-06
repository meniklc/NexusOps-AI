"""Recursive character text splitter.

A dependency-free re-implementation of the classic LangChain
`RecursiveCharacterTextSplitter` so it stays unit-testable offline. It tries a
descending list of separators (paragraph -> line -> sentence -> word -> char),
keeping chunks under `chunk_size` and overlapping consecutive chunks by
`chunk_overlap` characters to preserve context across boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class RecursiveCharacterTextSplitter:
    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: list[str] | None = None

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.separators is None:
            self.separators = list(DEFAULT_SEPARATORS)

    # -- public API --------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        splits = self._recursive_split(text, self.separators or DEFAULT_SEPARATORS)
        return self._merge(splits)

    # -- internals ---------------------------------------------------------
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Break text into pieces each <= chunk_size (best effort)."""
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        # Pick the first separator that actually occurs; fall back to char split.
        separator = separators[-1]
        remaining = separators[:]
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                remaining = []
                break
            if sep in text:
                separator = sep
                remaining = separators[i + 1 :]
                break

        if separator == "":
            # Hard character split as a last resort.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        pieces = text.split(separator)
        out: list[str] = []
        for piece in pieces:
            candidate = piece + separator
            if len(candidate) <= self.chunk_size:
                out.append(candidate)
            else:
                out.extend(self._recursive_split(candidate, remaining or [""]))
        return [p for p in out if p]

    def _merge(self, splits: list[str]) -> list[str]:
        """Greedily merge small splits into ~chunk_size chunks with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        length = 0

        for piece in splits:
            piece_len = len(piece)
            if length + piece_len > self.chunk_size and current:
                chunks.append("".join(current).strip())
                # Build overlap tail from the end of the current chunk.
                current, length = self._overlap_tail(current)
            current.append(piece)
            length += piece_len

        if current:
            tail = "".join(current).strip()
            if tail:
                chunks.append(tail)
        return [c for c in chunks if c]

    def _overlap_tail(self, current: list[str]) -> tuple[list[str], int]:
        if self.chunk_overlap <= 0:
            return [], 0
        tail: list[str] = []
        length = 0
        for piece in reversed(current):
            if length + len(piece) > self.chunk_overlap:
                break
            tail.insert(0, piece)
            length += len(piece)
        return tail, length


def split_text(
    text: str, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[str]:
    """Functional convenience wrapper."""
    return RecursiveCharacterTextSplitter(chunk_size, chunk_overlap).split_text(text)
