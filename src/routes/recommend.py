from fastapi import APIRouter, Depends
from typing import List, Dict, Any
from src.models.schemas import RecommendKeywordsRequest, RecommendKeywordsResponse, Suggestion, WeakSection
from src.services.agent import agent
from src.services.llm import count_tokens
from src.utils.security import api_key_required
import json
import os

# For demo "past data"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "past_blogs.json")

router = APIRouter()

def load_past() -> List[str]:
    import json, os
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@router.post("/recommend-keywords", response_model=RecommendKeywordsResponse, dependencies=[Depends(api_key_required)])
def recommend_keywords(req: RecommendKeywordsRequest):
    past_texts = load_past()
    init_state: Dict[str, Any] = {
        "past_texts": past_texts,
        "draft": req.draft,
        "user_profile": req.user_profile or {},
        "cursor_context": req.cursor_context or ""
    }

    out = agent.invoke(init_state)

    refined = out.get("refined_suggestions", []) or []
    # Normalize to Suggestion objects
    suggestions: List[Suggestion] = []
    for i, s in enumerate(refined):
        if isinstance(s, dict):
            phrase = s.get("phrase") or s.get("text") or str(s)
            why = s.get("why", "refined by agent")
        else:
            phrase, why = str(s), "refined by agent"
        suggestions.append(Suggestion(phrase=phrase, rank=i+1, why=why))

    weak_sections = [WeakSection(**ws) for ws in out.get("weak_sections", [])] if out.get("weak_sections") else []

    scores = out.get("scores", {})
    llm_usage = out.get("llm_token_usage", {"prompt": 0, "completion": 0, "total": 0})

    # Readability from scores map if needed; else compute minimal
    readability = scores.get("readability_norm", 0.0)
    relevance = scores.get("keyword_relevance", 0.0)
    final_score = scores.get("final_score", 0)

    # Estimated token usage (include user draft tokens)
    est_prompt = count_tokens(req.draft) + llm_usage.get("prompt", 0)
    est_completion = llm_usage.get("completion", 0)

    return RecommendKeywordsResponse(
        suggestions=suggestions,
        weak_sections=weak_sections,
        readability=round(readability, 2),
        relevance=round(relevance/100.0, 2),  # expose relevance as 0-1 in this field
        final_score=int(final_score),
        token_usage={"prompt": est_prompt, "completion": est_completion, "total": est_prompt + est_completion},
        estimated_token_cost={"input_tokens": est_prompt, "output_tokens": est_completion},
        warnings=[]
    )
