import os

from multiagent_rag.agents.base_ingestor import BaseIngestor
from multiagent_rag.utils.chunker import Chunker
from multiagent_rag.utils.ocr import read_pdf_with_easyocr


class PDFIngestor(BaseIngestor):
    def __init__(self):
        self.chunker = Chunker()

    def process(self, file_path: str):
        raw_text = read_pdf_with_easyocr(file_path)

        if not raw_text.strip():
            return []

        metadata = {
            "source": os.path.basename(file_path),
            "type": "pdf"
        }

        chunks = self.chunker.split_text(raw_text, metadata)

        return chunks
