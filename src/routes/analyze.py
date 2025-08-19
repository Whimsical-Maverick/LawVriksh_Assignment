from fastapi import APIRouter, Depends
from typing import List
from src.models.schemas import AnalyzeBlogsRequest, AnalyzeBlogsResponse, BlogAnalysisItem
from src.services.nlp import sentiment_scores, extract_topics_and_keywords, readability_score
from src.services.llm import count_tokens
from src.utils.security import api_key_required

router = APIRouter()

@router.post("/analyze-blogs", response_model=AnalyzeBlogsResponse, dependencies=[Depends(api_key_required)])
def analyze_blogs(req: AnalyzeBlogsRequest):
    texts: List[str] = req.blogs or []
    corpus = extract_topics_and_keywords(texts, top_k=8)

    results: List[BlogAnalysisItem] = []
    for t in texts:
        sent = sentiment_scores(t)
        read = readability_score(t)
        # Seed initial keywords from corpus + local TF-IDF (kept simple)
        initial_kw = corpus["keywords"][:5]
        token_usage = {"prompt": count_tokens(t), "completion": 0, "total": count_tokens(t)}
        results.append(BlogAnalysisItem(
            sentiment=sent,
            topics=corpus["topics"][:5],
            suggested_keywords=initial_kw,
            readability=round(read, 2),
            token_usage=token_usage
        ))

    return AnalyzeBlogsResponse(
        results=results,
        corpus_summary={"top_topics": corpus["topics"], "common_keywords": corpus["keywords"]}
    )