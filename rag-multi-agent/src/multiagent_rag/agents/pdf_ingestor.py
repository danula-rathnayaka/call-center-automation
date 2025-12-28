import os

import easyocr
import numpy as np
from multiagent_rag.utils.poppler import get_poppler_path
from pdf2image import convert_from_path


def read_pdf_with_easyocr(pdf_path):
    reader = easyocr.Reader(['en'])
    poppler_path = get_poppler_path()

    images = convert_from_path(
        pdf_path,
        poppler_path=poppler_path
    )

    full_text = ""

    for i, image in enumerate(images):
        print(f"Scanning Page {i + 1}...")
        image_np = np.array(image)
        result = reader.readtext(image_np, detail=0)
        full_text += f"\n--- Page {i + 1} ---\n{' '.join(result)}\n"

    return full_text


if __name__ == "__main__":
    print(read_pdf_with_easyocr(os.path.join("..", "..", "..", "data", "LankaLink_Customer_Support_Manual.pdf")))
