# models.py
from dataclasses import dataclass, field
from typing import List, Tuple
from text_similarity import (
            get_fuzzy_score,
            batch_semantic_scores,
            DEFAULT_THRESHOLDS
        )


@dataclass
class Line:
    content: str
    polygon: List[float]   # normalized polygon


@dataclass
class Page:
    page_number: int
    width: float
    height: float
    lines: List[Line] = field(default_factory=list)
    unit: str = "px"
    angle: float = 0.0


@dataclass
class Document:
    content: str
    pages: List[Page] = field(default_factory=list)

    def merge_polygons(self, page: Page, start: int, end: int) -> List[float]:
        # expand the selected region slightly (include one above/below)
        padded_start = max(0, start - 1)
        padded_end = min(len(page.lines), end + 1)

        pts = []
        for i in range(padded_start, padded_end):
            poly = page.lines[i].polygon
            for j in range(0, len(poly), 2):
                pts.append((poly[j], poly[j+1]))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        return [
            min(xs), min(ys),
            max(xs), min(ys),
            max(xs), max(ys),
            min(xs), max(ys),
        ]

    def find_text(self, target, similarity, threshold, page_number=None):
        

        if threshold is None:
            threshold = DEFAULT_THRESHOLDS.get(similarity, 70)

        pages = (
            [p for p in self.pages if p.page_number == page_number]
            if page_number else self.pages
        )

        # build search chunks
        candidate_texts = []
        meta = []  # (page, start, end)

        for page in pages:
            n = len(page.lines)
            for ws in range(1, 4):
                for start in range(n - ws + 1):
                    window = page.lines[start:start+ws]
                    joined = " ".join(l.content for l in window)

                    candidate_texts.append(joined)
                    meta.append((page, start, start+ws, joined))

        # compute scores
        if similarity == "semantic":
            scores = batch_semantic_scores(target, candidate_texts)
        else:
            scores = [get_fuzzy_score(target, c) for c in candidate_texts]

        # best match
        best = None
        for score, m in zip(scores, meta):
            page, s, e, joined = m
            if score >= threshold:
                if not best or score > best["score"]:
                    best = {
                        "page": page.page_number,
                        "start": s,
                        "end": e,
                        "text": joined,
                        "score": score,
                        "polygon": self.merge_polygons(page, s, e)
                    }

        return best
