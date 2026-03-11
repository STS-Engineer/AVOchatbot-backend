"""
File analysis service for uploaded user documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import csv

from loguru import logger

from app.core.config import settings
from app.services.llm import get_llm_service

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class FileAnalysisService:
    """Extracts text from uploaded files and asks LLM to analyze it."""

    SUPPORTED_EXTENSIONS = {".txt", ".csv", ".pdf", ".docx", ".xls", ".xlsx"}

    def __init__(self):
        self.llm = get_llm_service()

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract text and run a concise LLM analysis."""
        if not file_path.exists() or not file_path.is_file():
            return {
                "readable": False,
                "error": "File not found",
                "analysis": None,
                "extracted_chars": 0,
            }

        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            return {
                "readable": False,
                "error": f"Unsupported file type: {extension}",
                "analysis": None,
                "extracted_chars": 0,
            }

        extracted_text = self._extract_text(file_path)
        if not extracted_text.strip():
            return {
                "readable": False,
                "error": "No readable text found in file",
                "analysis": None,
                "extracted_chars": 0,
            }

        analysis = self.llm.summarize_uploaded_text(file_path.name, extracted_text)
        return {
            "readable": True,
            "error": None,
            "analysis": analysis,
            "extracted_chars": len(extracted_text),
            "preview": extracted_text[:1200],
        }

    def build_chat_file_context(self, uploaded_files: List[str], user_message: str) -> str:
        """Build a context block from uploaded files for chat completion."""
        if not uploaded_files:
            return ""

        uploads_dir = settings.uploads_dir_path
        contexts: List[str] = []

        logger.info(f"[CHAT-FILE] Looking for uploaded files in: {uploads_dir}")
        for raw_path in uploaded_files:
            safe_name = Path(str(raw_path).replace("\\", "/")).name
            candidate = uploads_dir / safe_name
            logger.info(f"[CHAT-FILE] Attempting to build context for: raw='{raw_path}', safe='{safe_name}', full_path='{candidate}'")

            if not candidate.exists() or not candidate.is_file():
                logger.warning(f"[CHAT-FILE] File not found or not a file: {candidate} (exists: {candidate.exists()})")
                continue

            extracted = self._extract_text(candidate)
            if not extracted.strip():
                logger.warning(f"Uploaded file had no readable text for chat context: {candidate.name}")
                continue

            # Use light summarization to control token usage.
            analysis = self.llm.summarize_uploaded_text(candidate.name, extracted)
            contexts.append(
                "\n".join(
                    [
                        f"FILE: {candidate.name}",
                        "FILE ANALYSIS:",
                        analysis or "No analysis produced.",
                        "FILE TEXT PREVIEW:",
                        extracted[:2500],
                    ]
                )
            )
            logger.info(f"Built chat file context for {candidate.name} ({len(extracted)} chars extracted)")

        if not contexts:
            return ""

        return "\n\n".join([
            "UPLOADED FILE CONTEXT:",
            "=" * 40,
            "\n\n".join(contexts),
            "=" * 40,
            f"User question about files: {user_message}",
        ])

    def _extract_text(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        try:
            if ext == ".txt":
                return file_path.read_text(encoding="utf-8", errors="ignore")

            if ext == ".csv":
                with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                    reader = csv.reader(handle)
                    rows = []
                    for idx, row in enumerate(reader):
                        rows.append(" | ".join(row))
                        if idx >= 200:
                            break
                return "\n".join(rows)

            if ext == ".pdf":
                if not PDF_AVAILABLE:
                    logger.warning("pypdf not installed; cannot parse PDF")
                    return ""
                reader = PdfReader(str(file_path))
                chunks: List[str] = []
                for page in reader.pages[:25]:
                    chunks.append(page.extract_text() or "")
                return "\n".join(chunks)

            if ext == ".docx":
                if not DOCX_AVAILABLE:
                    logger.warning("python-docx not installed; cannot parse DOCX")
                    return ""
                doc = docx.Document(str(file_path))
                return "\n".join([p.text for p in doc.paragraphs if p.text])

            if ext in {".xls", ".xlsx"}:
                if not PANDAS_AVAILABLE:
                    logger.warning("pandas not installed; cannot parse spreadsheet")
                    return ""
                sheets = pd.read_excel(str(file_path), sheet_name=None)
                lines: List[str] = []
                for sheet_name, df in sheets.items():
                    lines.append(f"Sheet: {sheet_name}")
                    lines.append(df.head(100).to_csv(index=False))
                return "\n".join(lines)

            return ""
        except Exception as exc:
            logger.warning(f"Failed to extract text from {file_path.name}: {exc}")
            return ""


_file_analysis_service: Optional[FileAnalysisService] = None


def get_file_analysis_service() -> FileAnalysisService:
    global _file_analysis_service
    if _file_analysis_service is None:
        _file_analysis_service = FileAnalysisService()
    return _file_analysis_service
