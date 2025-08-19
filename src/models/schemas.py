from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple

class AnalyzeBlogsRequest(BaseModel):
    blogs: List[str] = Field(..., description="Array of existing blog texts")

class BlogAnalysisItem(BaseModel):
    sentiment: Dict[str, float]
    topics: List[str]
    suggested_keywords: List[str]
    readability: float
    token_usage: Dict[str, int]

class AnalyzeBlogsResponse(BaseModel):
    results: List[BlogAnalysisItem]
    corpus_summary: Dict[str, Any]

class RecommendKeywordsRequest(BaseModel):
    draft: str
    cursor_context: Optional[str] = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)

class Suggestion(BaseModel):
    phrase: str
    rank: int
    why: str

class WeakSection(BaseModel):
    span: Tuple[int, int]
    issue: str
    fix_hint: str

class RecommendKeywordsResponse(BaseModel):
    suggestions: List[Suggestion]
    weak_sections: List[WeakSection]
    readability: float
    relevance: float
    final_score: int
    token_usage: Dict[str, int]
    estimated_token_cost: Dict[str, int]
    warnings: Optional[List[str]] = []
