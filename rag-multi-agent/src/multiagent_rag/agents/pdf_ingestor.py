import os

import easyocr
import numpy as np
from pdf2image import convert_from_path

from multiagent_rag.agents.base_ingestor import BaseIngestor
from multiagent_rag.utils.chunker import Chunker
from multiagent_rag.utils.poppler import get_poppler_path
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

class PDFIngestor(BaseIngestor):
    def __init__(self):
        self.chunker = Chunker()

    def process(self, file_path: str):
        raw_text = self._read_pdf_with_easyocr(file_path)

        if not raw_text.strip():
            return []

        metadata = {
            "source": os.path.basename(file_path),
            "type": "pdf"
        }

        chunks = self.chunker.split_text(raw_text, metadata)

        return chunks

    def _read_pdf_with_easyocr(self, pdf_path):
        reader = easyocr.Reader(['en'])
        poppler_path = get_poppler_path()

        try:
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
        except Exception as e:
            logger.error(f"PDF context conversion failed for {pdf_path}: {str(e)}")
            return ""

        full_text = ""
        for i, image in enumerate(images):
            image_np = np.array(image)
            result = reader.readtext(image_np, detail=0)
            page_text = ' '.join(result)
            full_text += f"\n{page_text}\n"

        return full_text
