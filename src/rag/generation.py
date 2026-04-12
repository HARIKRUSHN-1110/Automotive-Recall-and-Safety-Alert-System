"""
generation.py: LLM answer generation using Groq API.
Updated for CRAG — accepts both internal ChromaDB docs and
Tavily web search results, builds a combined prompt that
"""

import os
import logging
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from .retrieval import RetrievedDocument

load_dotenv()
logger = logging.getLogger(__name__)

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024

# Groq client singleton
_groq_client: Optional[Groq] = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. "
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

# System prompt
SYSTEM_PROMPT = """You are an automotive safety expert assistant for a \
recall prediction system powered by NHTSA complaint data.

Your rules:
- Answer based on the provided documents — internal NHTSA data takes priority
- If web sources are provided, use them to supplement or fill gaps
- Always clearly distinguish between official NHTSA data and web sources
- Highlight serious issues (crashes, fires, injuries, deaths) prominently
- If an official recall exists, always mention it clearly
- If the vehicle is not in the NHTSA database, explain this honestly and
  use web sources to still provide useful information
- Never speculate beyond what the documents say
- Keep answers concise but complete (around 5-6 paragraphs maximum)
- Use plain text, if needed then use markdown
- Use bullet points, if needed then use numbered lists
- Use markdown tables, if needed then use text tables
- if user is asking about a comparision between vehicles, then compare them based on NHTSA data and web resouces as well
  but prioritize NHTSA data if it is available.
- if user is asking for a recommendation then get personal with himself
  e.g. if i were you i would buy this car or i would not buy this car and also try to give strong reasons
"""

# Context builders

def _build_internal_context(documents: list[RetrievedDocument]) -> str:
    """Formats ChromaDB docs for the prompt."""
    if not documents:
        return "No relevant records found in the NHTSA database."

    recalls = [d for d in documents if d.doc_type == "recall"]
    complaints = [d for d in documents if d.doc_type == "complaint"]
    sections = []

    if recalls:
        sections.append("=== OFFICIAL NHTSA RECALL RECORDS ===")
        for i, doc in enumerate(recalls, 1):
            sections.append(
                f"\nRecall {i} (relevance: {doc.score:.2f}):"
            )
            sections.append(doc.to_context_string())

    if complaints:
        sections.append("\n=== NHTSA CONSUMER COMPLAINT REPORTS ===")
        for i, doc in enumerate(complaints, 1):
            sections.append(
                f"\nComplaint {i} (relevance: {doc.score:.2f}):"
            )
            sections.append(doc.to_context_string())

    return "\n".join(sections)

def _build_web_context(web_results: list) -> str:
    """Formats Tavily web results for the prompt."""
    if not web_results:
        return ""

    sections = ["\n=== WEB SEARCH RESULTS (via Tavily) ==="]
    for i, r in enumerate(web_results, 1):
        # Handle both WebSearchResult objects and dicts
        if isinstance(r, dict):
            title   = r.get("title", "")
            url     = r.get("url", "")
            snippet = r.get("snippet", "")
        else:
            title   = r.title
            url     = r.url
            snippet = r.snippet

        sections.append(f"\nWeb Result {i}:")
        sections.append(
            f"[WEB SOURCE] {title}\n"
            f"URL: {url}\n"
            f"Content: {snippet}"
        )

    return "\n".join(sections)

def _build_user_prompt(
    query: str,
    internal_context: str,
    web_context: str,
    quality: str,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """
    Builds the full prompt with quality instructions.
    """
    vehicle_str = ""
    if any([make, model, year]):
        parts = [
            p for p in [make, model, str(year) if year else None] if p
        ]
        vehicle_str = f"Vehicle context: {' '.join(parts)}\n\n"

    if quality == "sufficient":
        instruction = (
            "Answer using ONLY the NHTSA documents below. "
            "Do not use outside knowledge."
        )
    elif quality == "partial":
        instruction = (
            "The NHTSA database has limited information on this topic. "
            "Use NHTSA data as your primary source and supplement with "
            "web search results where NHTSA data is insufficient. "
            "Clearly indicate when drawing from web sources."
        )
    else:
        instruction = (
            "This vehicle or topic was NOT found in the NHTSA database. "
            "Answer using the web search results below. "
            "Clearly state this information comes from web sources, "
            "not official NHTSA records."
        )

    combined_context = internal_context
    if web_context:
        combined_context += "\n" + web_context

    return (
        f"{vehicle_str}"
        f"INSTRUCTION: {instruction}\n\n"
        f"DOCUMENTS:\n{combined_context}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Answer:"
    )

# Main generation function

def generate_answer(
    query: str,
    documents: list[RetrievedDocument],
    web_results: Optional[list] = None,
    quality: str = "sufficient",
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Generates a answer.

    Args:
        query:                User's question
        documents:            ChromaDB retrieved docs
        web_results:          Tavily web results (may be empty list)
        quality:              "sufficient" or "partial" or "insufficient"
        make/model/year:      Vehicle context
        conversation_history: Previous turns for multi-turn chat

    Returns:
        Generated answer string.
    """
    web_results = web_results or []

    if not documents and not web_results:
        return (
            "Couldn't find any relevant information in the NHTSA database "
            "or via web search for your question. "
            "Please try rephrasing, or check the vehicle spelling."
        )

    try:
        client = get_groq_client()

        internal_context = _build_internal_context(documents)
        web_context = _build_web_context(web_results)

        user_prompt = _build_user_prompt(
            query=query,
            internal_context=internal_context,
            web_context=web_context,
            quality=quality,
            make=make,
            model=model,
            year=year,
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history[-6:])

        messages.append({"role": "user", "content": user_prompt})

        logger.info(
            f"Groq | quality={quality} | "
            f"internal={len(documents)} | web={len(web_results)}"
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )

        answer = response.choices[0].message.content.strip()
        logger.info("Groq response received.")
        return answer

    except ValueError as e:
        logger.error(f"Groq config error: {e}")
        return f"Configuration error: {str(e)}"
    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        return f"Error generating answer: {str(e)}. Please try again."

# Legacy source summary

def summarise_sources(documents: list[RetrievedDocument]) -> str:
    """Legacy — rag_engine._build_source_summary is preferred."""
    if not documents:
        return "No sources found."
    n_complaints = sum(1 for d in documents if d.doc_type == "complaint")
    n_recalls = sum(1 for d in documents if d.doc_type == "recall")
    parts = []
    if n_complaints:
        parts.append(
            f"{n_complaints} complaint{'s' if n_complaints > 1 else ''}"
        )
    if n_recalls:
        parts.append(
            f"{n_recalls} recall{'s' if n_recalls > 1 else ''}"
        )
    return f"Based on {' and '.join(parts)}." if parts else "No sources."