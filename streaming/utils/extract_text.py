from __future__ import annotations

import io
import os
from typing import Optional


def _extract_pdf(content: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text as _pdf_extract  # type: ignore
        with io.BytesIO(content) as fh:
            return _pdf_extract(fh) or ''
    except Exception:
        return ''


def _extract_docx(content: bytes) -> str:
    try:
        # python-docx
        import docx  # type: ignore
        with io.BytesIO(content) as fh:
            doc = docx.Document(fh)
            return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception:
        try:
            # docx2txt fallback
            import tempfile, docx2txt  # type: ignore
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=True) as tmp:
                tmp.write(content)
                tmp.flush()
                return docx2txt.process(tmp.name) or ''
        except Exception:
            return ''


def _extract_txt(content: bytes) -> str:
    try:
        return content.decode('utf-8', errors='replace')
    except Exception:
        return ''


def _extract_md(content: bytes) -> str:
    return _extract_txt(content)


def _extract_doc(content: bytes) -> str:
    # Legacy .doc support is limited without native deps; return empty string.
    # Users can convert to .docx for richer extraction.
    return ''


def extract_text_from_bytes(path: str, content: bytes) -> str:
    """Heuristic text extraction from common document formats.

    Supports: .pdf, .docx, .txt, .md (best-effort for .doc returns empty string).
    """
    ext = os.path.splitext(path.lower())[1]
    if ext == '.pdf':
        return _extract_pdf(content)
    if ext == '.docx':
        return _extract_docx(content)
    if ext == '.doc':
        return _extract_doc(content)
    if ext in ('.txt', '.log', '.cfg', '.ini'):
        return _extract_txt(content)
    if ext == '.md':
        return _extract_md(content)
    # Fallback to utf-8 decode
    return _extract_txt(content)

