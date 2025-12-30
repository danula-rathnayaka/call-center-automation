import easyocr
import numpy as np
from pdf2image import convert_from_path

from multiagent_rag.utils.poppler import get_poppler_path


def read_pdf_with_easyocr(pdf_path):
    reader = easyocr.Reader(['en'])
    poppler_path = get_poppler_path()

    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    except Exception:
        return ""

    full_text = ""
    for i, image in enumerate(images):
        image_np = np.array(image)
        result = reader.readtext(image_np, detail=0)
        page_text = ' '.join(result)
        full_text += f"\n{page_text}\n"

    return full_text
