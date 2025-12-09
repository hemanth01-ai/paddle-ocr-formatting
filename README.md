#Readme

This project provides a  pipeline to:

1. Convert a PDF into images  
2. Run PaddleOCR on each page  
3. Build a Document → Page → Line structure  
4. Search text using fuzzy or semantic similarity  
5. Return matched text with merged bounding box

---

## Installation

```bash
pip install fuzzywuzzy 
pip install paddlepaddle==3.2.0
pip install "paddleocr[all]"



## Basic Usage

```bash
from pipeline import run_paddle_ocr_on_pdf

doc = run_paddle_ocr_on_pdf("input.pdf")

result = doc.find_text(
    target="ARTHROPLASTY KNEE",
    similarity="fuzzy",   # or "semantic"
    threshold=50
)

print(result)


# Example output:

```bash
{
  "page": 1,
  "start": 67,
  "end": 69,
  "text": "ASevere arthritis left knee Complications",
  "score": 53.0,
  "polygon": [np.float64(0.03627450980392157), np.float64(0.7674242424242425), np.float64(0.2568627450980392), np.float64(0.7674242424242425), np.float64(0.2568627450980392), np.float64(0.825), np.float64(0.03627450980392157), np.float64(0.825)]
}

