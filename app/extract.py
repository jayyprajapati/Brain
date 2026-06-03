"""Plain-text extraction from uploaded documents.

A generic capability: turn an uploaded file's bytes into text. Brain stays
content-agnostic — it has no idea whether the document is a resume, a contract,
or a recipe. PDF and DOCX are first-class; legacy ``.doc`` is not supported.
"""
from __future__ import annotations

import io
import re

_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


class ExtractionError(Exception):
    """Raised when a document cannot be parsed into text."""


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError("pypdf is not installed") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"could not read PDF: {exc}") from exc
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — skip an unreadable page, keep the rest
            continue
    return "\n\n".join(p for p in pages if p.strip())


def _extract_docx(data: bytes) -> str:
    try:
        import docx  # python-docx
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError("python-docx is not installed") from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"could not read DOCX: {exc}") from exc
    parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    # Pull table cell text too — resumes often lay out sections in tables.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" • ".join(cells))
    return "\n".join(parts)


def extract_text(data: bytes, *, filename: str = "", content_type: str = "") -> str:
    """Extract plain text from ``data``, dispatching on extension / MIME type."""
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    if name.endswith(".pdf") or "pdf" in ctype:
        text = _extract_pdf(data)
    elif name.endswith(".docx") or "openxmlformats-officedocument.wordprocessingml" in ctype:
        text = _extract_docx(data)
    elif name.endswith(".doc") or ctype == "application/msword":
        raise ExtractionError("legacy .doc files are not supported — please upload a PDF or DOCX")
    elif name.endswith(".txt") or ctype.startswith("text/"):
        text = data.decode("utf-8", errors="ignore")
    else:
        # Last-ditch attempt: treat as UTF-8 text.
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExtractionError(
                f"unsupported file type (filename='{filename}', content_type='{content_type}')"
            ) from exc

    cleaned = _clean(text)
    if not cleaned:
        raise ExtractionError("no extractable text found in document")
    return cleaned
