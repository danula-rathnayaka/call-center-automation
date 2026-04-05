import hashlib
import os

import easyocr
import numpy as np
from pdf2image import convert_from_path

from multiagent_rag.agents.base_ingestor import BaseIngestor
from multiagent_rag.utils.chunker import Chunker
from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.poppler import get_poppler_path

logger = get_logger(__name__)


class PDFIngestor(BaseIngestor):
    def __init__(self):
        self.chunker = Chunker()

    def process(self, file_path: str):
        raw_text = self._extract_text(file_path)

        if not raw_text.strip():
            return [], ""

        doc_hash = self._compute_hash(file_path)
        metadata = {"source": os.path.basename(file_path), "type": "pdf", "document_hash": doc_hash, }
        chunks = self.chunker.split_text(raw_text, metadata)
        return chunks, doc_hash

    def _extract_text(self, file_path: str) -> str:
        text = self._try_pdfplumber(file_path)
        if text.strip():
            logger.info(f"Extracted text from digital PDF: {file_path}")
            return text
        logger.info(f"Digital extraction empty, falling back to OCR: {file_path}")
        return self._read_pdf_with_easyocr(file_path)

    def _try_pdfplumber(self, file_path: str) -> str:
        try:
            import pdfplumber
            full_text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(page_text)
            return "\n".join(full_text)
        except ImportError:
            logger.warning("pdfplumber not installed, skipping digital extraction")
            return ""
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return ""

    def _read_pdf_with_easyocr(self, pdf_path: str) -> str:
        reader = easyocr.Reader(["en"])
        poppler_path = get_poppler_path()
        try:
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
        except Exception as e:
            logger.error(f"PDF conversion failed for {pdf_path}: {e}")
            return ""

        full_text = ""
        for image in images:
            image_np = np.array(image)
            result = reader.readtext(image_np, detail=0)
            full_text += f"\n{' '.join(result)}\n"
        return full_text

    def _compute_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()
