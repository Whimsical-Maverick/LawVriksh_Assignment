from typing import Dict, List
from .nlp import readability_score, tfidf_similarity

def clamp_0_100(x: float) -> float:
    return max(0.0, min(100.0, x))

def compute_scores(draft: str, suggested_phrases: List[str], user_profile: Dict) -> Dict[str, float]:
    # Keyword relevance: average TF-IDF sim of draft vs phrase list joined
    phrases_text = " ".join(suggested_phrases or [])
    rel = tfidf_similarity(draft or "", phrases_text or "")
    keyword_relevance = clamp_0_100(rel * 100)

    # Readability normalized
    fre = readability_score(draft or "")
    readability_norm = clamp_0_100(fre)

    # Profile alignment: compare draft to preferred topics text
    preferred = user_profile.get("preferred_topics", [])
    prefs_text = " ".join(preferred) if isinstance(preferred, list) else str(preferred)
    palign = tfidf_similarity(draft or "", prefs_text or "")
    profile_alignment = clamp_0_100(palign * 100)

    final = 0.4 * keyword_relevance + 0.3 * readability_norm + 0.3 * profile_alignment

    return {
        "keyword_relevance": round(keyword_relevance, 2),
        "readability_norm": round(readability_norm, 2),
        "profile_alignment": round(profile_alignment, 2),
        "final_score": int(round(final))
    }
