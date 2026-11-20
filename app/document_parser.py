from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .errors import AppError


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".json", ".html", ".htm"}
MACRO_EXTENSIONS = {".docm", ".pptm", ".xlsm"}
MAX_UNCOMPRESSED_OFFICE_BYTES = 40 * 1024 * 1024
MAX_EXTRACTED_CHARS = 300_000


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AppError("TEXT_ENCODING_UNSUPPORTED", 422)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\u00a0]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()[:MAX_EXTRACTED_CHARS]


def _validate_zip(data: bytes) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise AppError("OFFICE_FILE_DAMAGED", 422) from exc
    total = sum(item.file_size for item in archive.infolist())
    if total > MAX_UNCOMPRESSED_OFFICE_BYTES:
        archive.close()
        raise AppError("OFFICE_FILE_TOO_LARGE", 413)
    if len(data) and total / len(data) > 120:
        archive.close()
        raise AppError("OFFICE_COMPRESSION_UNSAFE", 422)
    names = {item.filename.lower() for item in archive.infolist()}
    if any("vbaproject.bin" in name or name.endswith(".exe") for name in names):
        archive.close()
        raise AppError("ACTIVE_CONTENT_REJECTED", 422)
    return archive


def _extract_docx(data: bytes) -> str:
    with _validate_zip(data) as archive:
        names = set(archive.namelist())
        if "word/document.xml" not in names:
            raise AppError("FILE_TYPE_MISMATCH", 422)
        root = ElementTree.fromstring(archive.read("word/document.xml"))
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs: list[str] = []
        for paragraph in root.iter(f"{namespace}p"):
            words = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
            if words:
                paragraphs.append("".join(words))
        return "\n\n".join(paragraphs)


def _slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _extract_pptx(data: bytes) -> str:
    with _validate_zip(data) as archive:
        slides = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=_slide_number,
        )
        if not slides:
            raise AppError("FILE_TYPE_MISMATCH", 422)
        parts: list[str] = []
        drawing_namespace = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        for name in slides:
            number = _slide_number(name)
            root = ElementTree.fromstring(archive.read(name))
            words = [node.text or "" for node in root.iter(f"{drawing_namespace}t")]
            parts.append(f"[[Slide {number}]]\n" + "\n".join(words))
        return "\n\n".join(parts)


def _extract_pdf(data: bytes) -> str:
    executable = shutil.which("pdftotext")
    if not executable:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise AppError("PDF_PARSER_UNAVAILABLE", 503) from exc
        reader = PdfReader(io.BytesIO(data))
        return "\f".join(page.extract_text() or "" for page in reader.pages)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(data)
        handle.flush()
        try:
            process = subprocess.run(
                [executable, "-layout", handle.name, "-"],
                capture_output=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppError("PDF_PARSE_TIMEOUT", 422) from exc
    if process.returncode != 0:
        raise AppError("PDF_PARSE_FAILED", 422)
    return _decode_text(process.stdout)


def _extract_csv(data: bytes) -> str:
    decoded = _decode_text(data)
    reader = csv.reader(io.StringIO(decoded))
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in reader)


def parse_document(filename: str, data: bytes) -> tuple[str, str, dict[str, Any]]:

    safe_name = Path(filename).name[:240]
    extension = Path(safe_name).suffix.lower()
    if extension in MACRO_EXTENSIONS:
        raise AppError("MACRO_FILE_REJECTED", 422)
    if extension not in ALLOWED_EXTENSIONS:
        raise AppError("UNSUPPORTED_FILE", 415, details={"extension": extension or "none"})
    if not data:
        raise AppError("EMPTY_FILE", 422)

    mime_type = "application/octet-stream"
    if extension == ".pdf":
        if not data.startswith(b"%PDF"):
            raise AppError("FILE_TYPE_MISMATCH", 422)
        text = _extract_pdf(data)
        mime_type = "application/pdf"
    elif extension == ".docx":
        if not data.startswith(b"PK"):
            raise AppError("FILE_TYPE_MISMATCH", 422)
        text = _extract_docx(data)
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif extension == ".pptx":
        if not data.startswith(b"PK"):
            raise AppError("FILE_TYPE_MISMATCH", 422)
        text = _extract_pptx(data)
        mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    elif extension in {".html", ".htm"}:
        parser = _TextHTMLParser()
        parser.feed(_decode_text(data))
        text = parser.text()
        mime_type = "text/html"
    elif extension == ".csv":
        text = _extract_csv(data)
        mime_type = "text/csv"
    else:
        text = _decode_text(data)
        mime_type = {
            ".md": "text/markdown",
            ".json": "application/json",
        }.get(extension, "text/plain")

    cleaned = _clean_text(text)
    if len(cleaned) < 20:
        raise AppError("DOCUMENT_TEXT_TOO_SHORT", 422)
    return cleaned, mime_type, {
        "parser": extension.lstrip("."),
        "characters": len(cleaned),
        "truncated": len(text) > MAX_EXTRACTED_CHARS,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def chunk_text(material_id: str, text: str, target_chars: int = 1_800) -> list[dict[str, Any]]:

    if not text:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    length = len(text)
    while start < length:
        tentative = min(start + target_chars, length)
        end = tentative
        if tentative < length:
            boundary = max(text.rfind("\n\n", start + 600, tentative), text.rfind("。", start + 600, tentative))
            if boundary > start:
                end = boundary + 1
        segment = text[start:end].strip()
        leading = len(text[start:end]) - len(text[start:end].lstrip())
        actual_start = start + leading
        if segment:
            chunks.append(
                {
                    "id": f"{material_id}:c{len(chunks) + 1}",
                    "material_id": material_id,
                    "text": segment,
                    "char_start": actual_start,
                    "char_end": actual_start + len(segment),
                    "locator": f"chars {actual_start}-{actual_start + len(segment)}",
                }
            )
        start = max(end, start + 1)
    return chunks
