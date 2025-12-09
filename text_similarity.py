# text_similarity.py
import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

semantic_model = None

DEFAULT_THRESHOLDS = {"fuzzy": 70, "semantic": 80}


def get_fuzzy_score(text1, text2):
    return float(fuzz.token_sort_ratio(text1, text2))


def batch_semantic_scores(target: str, candidates: list):
    global semantic_model
    if semantic_model is None:
        semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if not candidates:
        return []

    target_emb = semantic_model.encode([target])[0]
    cand_embs = semantic_model.encode(candidates)

    t_norm = target_emb / np.linalg.norm(target_emb)
    c_norms = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)

    cosine_scores = np.dot(c_norms, t_norm)
    return [(float(s) + 1) * 50 for s in cosine_scores]
