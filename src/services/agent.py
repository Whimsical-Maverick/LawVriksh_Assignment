from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from .nlp import extract_topics_and_keywords, sentiment_scores, readability_score, find_weak_spans
from .scoring import compute_scores
from .llm import llm_complete_json
from ..utils.retry import retry_backoff

# Graph state is a plain dict
def node_analyze_past(state: Dict[str, Any]) -> Dict[str, Any]:
    past_texts = state.get("past_texts", [])
    # Extract corpus topics/keywords
    tk = extract_topics_and_keywords(past_texts, top_k=8)
    # Basic corpus sentiment (avg)
    sentiments = [sentiment_scores(t) for t in past_texts] or [{"positive":0,"neutral":1,"negative":0}]
    avg_pos = sum(s["positive"] for s in sentiments)/len(sentiments)
    avg_neu = sum(s["neutral"] for s in sentiments)/len(sentiments)
    avg_neg = sum(s["negative"] for s in sentiments)/len(sentiments)
    return {"corpus_topics": tk["topics"], "corpus_keywords": tk["keywords"], "corpus_sentiment": {"positive":avg_pos,"neutral":avg_neu,"negative":avg_neg}}

def node_analyze_draft(state: Dict[str, Any]) -> Dict[str, Any]:
    draft = state.get("draft", "")
    weak = find_weak_spans(draft)
    read = readability_score(draft)
    # initial suggestions from draft gaps vs corpus keywords (very simple: reuse corpus keywords)
    suggestions = [{"phrase": kw, "rank": i+1, "why": "common in past corpus"} for i, kw in enumerate(state.get("corpus_keywords", [])[:5])]
    return {"weak_sections": weak, "readability": read, "baseline_suggestions": suggestions}

def node_llm_refine(state: Dict[str, Any]) -> Dict[str, Any]:
    draft = state.get("draft", "")
    topics = state.get("corpus_topics", [])
    profile = state.get("user_profile", {})
    baseline = state.get("baseline_suggestions", [])

    prompt = f"""
Task: Suggest 3-5 precise, inline-ready keywords/phrases to insert next in the draft; refine based on corpus topics and user profile.
Inputs:
- Draft: {draft}
- Corpus topics: {topics}
- Baseline suggestions: {baseline}
- Profile: {profile}
Output JSON with keys: suggestions (array of objects with phrase, rank, why).
"""
    out, usage = llm_complete_json(prompt)
    suggestions = out.get("suggestions") or baseline
    return {"refined_suggestions": suggestions, "llm_token_usage": usage}

def node_score(state: Dict[str, Any]) -> Dict[str, Any]:
    draft = state.get("draft", "")
    suggestions = [s["phrase"] if isinstance(s, dict) else str(s) for s in state.get("refined_suggestions", [])]
    profile = state.get("user_profile", {})
    comps = compute_scores(draft, suggestions, profile)
    return {"scores": comps}

# Build graph
def build_agent():
    g = StateGraph(dict)
    g.add_node("analyze_past", lambda s: retry_backoff(lambda: node_analyze_past(s)))
    g.add_node("analyze_draft", lambda s: retry_backoff(lambda: node_analyze_draft(s)))
    g.add_node("llm_refine",  lambda s: retry_backoff(lambda: node_llm_refine(s)))
    g.add_node("score",       lambda s: retry_backoff(lambda: node_score(s)))

    g.set_entry_point("analyze_past")
    g.add_edge("analyze_past", "analyze_draft")
    g.add_edge("analyze_draft", "llm_refine")
    g.add_edge("llm_refine", "score")
    g.add_edge("score", END)
    return g.compile()

agent = build_agent()
