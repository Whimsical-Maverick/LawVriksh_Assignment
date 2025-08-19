# LawVriksh_Assignment


# Agentic Blog Support System

## 1. Overview

React FE + FastAPI BE + LangGraph orchestrator. Provides blog analysis and real-time keyword recommendations.

## 2. Architecture

- FastAPI endpoints: /api/analyze-blogs, /api/recommend-keywords
- LangGraph nodes: analyze_past → analyze_draft → llm_refine → score
- NLP: VADER (sentiment), TF-IDF (topics/keywords), textstat (readability)
- LLM: OpenAI GPT-4o-mini (fallback heuristic if key missing)
- Security: API Key (optional JWT)

## 3. API Contracts

Document request/response shapes (see Postman + code).

## 4. Agentic Workflow (LangGraph)

- State carries: draft, user_profile, corpus_topics/keywords, weak_sections, scores
- Retries: up to 3 with exponential backoff on draft/LLM nodes

## 5. Model Choice & Prompts

- GPT-4o-mini for reasoning and JSON control
- Token optimization: compressed instruction with explicit JSON schema
- Before/after token counts shown from tiktoken and OpenAI usage

## 6. Scoring

final_score = 0.4*keyword_relevance + 0.3*readability_norm + 0.3*profile_alignment
All components scaled to [0,100]

## 7. Token Usage

- Counted per call; included in API responses

## 8. Security

- Header `X-API-Key` (env var)

## 9. Limitations & Future Work

- Replace TF-IDF with embeddings
- Add caching and streaming suggestions
- Better domain-specific sentiment/topic models
