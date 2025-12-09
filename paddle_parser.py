# paddle_parser.py
import json
from PIL import Image
from models import Document, Page, Line


def normalize_polygon(poly, w, h):
    out = []
    for x, y in poly:
        out.append(x / w)
        out.append(y / h)
    return out


class PaddleDocumentParser:
    """
    Parse a multi-page Paddle OCR output into a Document object.
    """

    def parse_pages(self, paddle_outputs: list) -> Document:
        """
        paddle_outputs:
        [
          {
            "page_number": 1,
            "img_path": "...",
            "rec_texts": [...],
            "rec_polys": [...]
          },
          ...
        ]
        """
        all_pages = []
        full_text_lines = []

        for page_data in paddle_outputs:
            page_num = page_data["page_number"]
            img_path = page_data["img_path"]

            img = Image.open(img_path)
            w, h = img.size

            rec_texts = page_data["rec_texts"]
            rec_polys = page_data["rec_polys"]

            page = Page(page_num, w, h)

            for text, poly in zip(rec_texts, rec_polys):
                full_text_lines.append(text)
                norm_poly = normalize_polygon(poly, w, h)
                page.lines.append(Line(text, norm_poly))

            all_pages.append(page)

        full_text = "\n".join(full_text_lines)
        return Document(content=full_text, pages=all_pages)
