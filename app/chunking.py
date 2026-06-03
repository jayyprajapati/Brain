"""Structure-aware semantic chunking.

The same function is used to chunk ingested documents and (trivially) the user
query. The strategy:

  1. Split the text into blocks on markdown headings / blank lines, tracking the
     current heading so it can travel with each chunk as metadata. Non-semantic
     noise is dropped up front: horizontal rules (``---``, ``***``, ``===``) are
     skipped entirely, and document-separator delimiters (``===== file.md =====``,
     the pattern apps use to concatenate many files into one ingest payload) are
     treated as document boundaries that RESET the heading — so a headingless
     section never inherits the previous file's heading.
  2. Sentence-split each block.
  3. Group consecutive sentences while they stay semantically similar (cosine of
     the running centroid) AND the chunk stays under a token budget. Start a new
     chunk at a semantic boundary or when the size cap is hit.
  4. Merge runt fragments forward and carry a 1-sentence overlap between
     neighbours so context isn't severed mid-thought.
  5. Drop chunks with no real content (below a word threshold and no URL) and
     collapse exact duplicates, then number the survivors contiguously — so no
     empty / separator / duplicate vectors ever reach the store.
  6. URLs are pulled out into metadata AND kept inline, so "give me the links"
     style questions can be answered exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import embeddings
from .config import settings

_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
# A horizontal rule: a line that is only a run of -, *, _ or = (markdown <hr>).
_HR_RE = re.compile(r"^\s*([-*_=])\1{2,}\s*$")
# A document-separator delimiter wrapping a label, e.g. "===== identity.md =====".
# Captures the label so it can become a readable heading for the section below.
_DOC_DELIM_RE = re.compile(r"^\s*={2,}\s*(\S.*?\S)\s*={2,}\s*$")
# Word tokens, for the meaningful-content guard.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Sentence boundary: end punctuation followed by whitespace + capital/quote/digit.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def _label_to_heading(label: str) -> str:
    """Turn a delimiter label like "retrieval_facts.md" into "Retrieval Facts"."""
    label = re.sub(r"\.[A-Za-z0-9]+$", "", label.strip())  # drop file extension
    label = re.sub(r"[_\-]+", " ", label).strip()
    return label.title() if label else ""


@dataclass
class Chunk:
    text: str
    heading: str = ""
    urls: list[str] = field(default_factory=list)
    chunk_index: int = 0


def _estimate_tokens(text: str) -> int:
    # Cheap, tokenizer-free heuristic: ~1.3 tokens per whitespace word.
    words = len(text.split())
    return int(words * 1.3) + 1


def _extract_urls(text: str) -> list[str]:
    seen: list[str] = []
    for match in _URL_RE.findall(text):
        url = match.rstrip(".,;:")
        if url not in seen:
            seen.append(url)
    return seen


def _split_blocks(text: str) -> list[tuple[str, str]]:
    """Yield (heading, block_text) pairs, carrying the latest heading downward."""
    blocks: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            blocks.append((current_heading, body))
        buffer.clear()

    for raw_line in text.splitlines():
        heading_match = _HEADING_RE.match(raw_line)
        if heading_match:
            flush()
            current_heading = heading_match.group(2).strip()
            continue
        # A document-separator delimiter ("===== file.md =====") ends the current
        # section and resets the heading, so a headingless file below it doesn't
        # inherit the previous file's heading. Checked before _HR_RE since a bare
        # "=====" with no label is just a horizontal rule.
        delim_match = _DOC_DELIM_RE.match(raw_line)
        if delim_match:
            flush()
            current_heading = _label_to_heading(delim_match.group(1))
            continue
        # Horizontal rules carry no meaning — drop them so they never become chunks.
        if _HR_RE.match(raw_line):
            flush()
            continue
        if not raw_line.strip():
            flush()
            continue
        buffer.append(raw_line.strip())
    flush()
    return blocks


def _split_sentences(block: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_RE.split(block) if s.strip()]
    return parts or ([block.strip()] if block.strip() else [])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _semantic_group(sentences: list[str]) -> list[list[int]]:
    """Group sentence indices into semantically coherent, size-bounded chunks."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return [[0]]

    vecs = embeddings.embed_documents(sentences)
    groups: list[list[int]] = []
    current: list[int] = [0]
    centroid = vecs[0].astype(float)
    current_tokens = _estimate_tokens(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine(centroid, vecs[i].astype(float))
        next_tokens = _estimate_tokens(sentences[i])
        over_cap = current_tokens + next_tokens > settings.chunk_max_tokens
        semantic_break = sim < settings.semantic_threshold

        if (semantic_break or over_cap) and current_tokens >= settings.chunk_min_tokens:
            groups.append(current)
            current = [i]
            centroid = vecs[i].astype(float)
            current_tokens = next_tokens
        else:
            current.append(i)
            centroid = (centroid * len(current) + vecs[i].astype(float)) / (len(current) + 1)
            current_tokens += next_tokens

    if current:
        # Merge a trailing runt into the previous group if one exists.
        if groups and current_tokens < settings.chunk_min_tokens:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _is_meaningful(text: str) -> bool:
    """True if a chunk is worth embedding: it has a URL or enough real words."""
    if _URL_RE.search(text):
        return True
    return len(_WORD_RE.findall(text)) >= settings.chunk_min_content_words


def _dedup_key(text: str) -> str:
    """Normalize for exact-duplicate detection (case- and whitespace-insensitive)."""
    return " ".join(text.lower().split())


def chunk_text(text: str) -> list[Chunk]:
    """Chunk a document into semantically coherent, overlapping pieces.

    Non-semantic junk (separators, stray punctuation) and exact duplicates are
    dropped before numbering, so chunk_index stays contiguous over real chunks.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for heading, block in _split_blocks(text):
        sentences = _split_sentences(block)
        groups = _semantic_group(sentences)
        prev_tail = ""  # 1-sentence overlap carried from the previous chunk
        for group in groups:
            body = " ".join(sentences[i] for i in group)
            full = f"{prev_tail} {body}".strip() if prev_tail else body
            prev_tail = sentences[group[-1]]
            if not _is_meaningful(full):
                continue
            key = _dedup_key(full)
            if key in seen:
                continue
            seen.add(key)
            chunks.append(
                Chunk(
                    text=full,
                    heading=heading,
                    urls=_extract_urls(full),
                    chunk_index=len(chunks),
                )
            )
    return chunks
