# pipeline.py
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path, pdfinfo_from_path
from paddleocr import PaddleOCR

from paddle_parser import PaddleDocumentParser

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en'
)


def pdf_to_images(pdf_path, output_dir):
    info = pdfinfo_from_path(pdf_path)
    pages = info["Pages"]

    def _convert(page):
        imgs = convert_from_path(pdf_path, dpi=120, first_page=page, last_page=page)
        path = os.path.join(output_dir, f"page_{page}.jpg")
        imgs[0].save(path, "JPEG")
        return page, path

    with ThreadPoolExecutor(max_workers=4) as exe:
        return dict(exe.map(_convert, range(1, pages + 1)))


def run_paddle_ocr_on_pdf(pdf_path: str) -> "Document":
    """
    Convert a full PDF → PaddleOCR → Document object.
    Images are stored in a temporary folder and deleted after processing.
    """
    parser = PaddleDocumentParser()

    with tempfile.TemporaryDirectory(prefix="pdf_images_") as tmpdir:
        img_dict = pdf_to_images(pdf_path, tmpdir)

        all_page_outputs = []

        for page_num, img_path in img_dict.items():
            result = ocr.predict(img_path)

            rec_texts = result[0]['rec_texts']
            rec_polys = result[0]['rec_polys']

            all_page_outputs.append({
                "page_number": page_num,
                "img_path": img_path,
                "rec_texts": rec_texts,
                "rec_polys": rec_polys,
            })

        # Parse while images still exist
        document = parser.parse_pages(all_page_outputs)

    # Temporary folder is deleted after this point
    return document
