"""Text extraction, NER, and normalization of resumes and job descriptions into Profiles."""
from collections.abc import Iterable
from pathlib import Path

import pymupdf


def read_text(data: bytes, filename: str) -> str:
    """Text of a .pdf or .txt file. ValueError if the type is unsupported or there is no text."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            text = "".join(page.get_text() for page in pdf)
    elif ext == ".txt":
        text = data.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError(f"unsupported file type '{ext or 'none'}' (use .pdf or .txt)")
    if not text.strip():
        raise ValueError("no extractable text (scanned PDF?)")
    return text


def read_all(items: Iterable[tuple[str, bytes]]) -> tuple[list[tuple[str, str]], list[str]]:
    """Read every (filename, bytes); bad files become warnings instead of errors."""
    docs, warnings = [], []
    for name, data in items:
        try:
            docs.append((name, read_text(data, name)))
        except Exception as e:  # corrupt PDFs raise various pymupdf errors
            warnings.append(f"skipped {name}: {e}")
    return docs, warnings
