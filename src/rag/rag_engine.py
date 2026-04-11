"""
rag_engine.py: CRAG (Corrective RAG) pipeline combining internal NHTSA data
and Tavily web search.

Quality criteria:
    sufficient   : ChromaDB only (top score >= 0.5, 2+ relevant docs)
    partial      : ChromaDB + web (some relevant docs but not enough)
    insufficient : web only (vehicle not in NHTSA database)
"""

import logging
from typing import Optional

from .retrieval import retrieve_all, RetrievedDocument, DEFAULT_TOP_K
from .generation import generate_answer, summarise_sources
from .embeddings import get_index_status
from .web_search import search_web, filter_relevant_results, WebSearchResult

logger = logging.getLogger(__name__)

# Confidence thresholds
SUFFICIENT_SCORE = 0.55   # top doc above this + 2+ relevant = internal only
PARTIAL_SCORE = 0.45      # top doc above this = blend internal + web
MIN_RELEVANT_DOCS = 2     # minimum docs above PARTIAL_SCORE for "sufficient"

# Quality evaluator
def evaluate_retrieval_quality(
    docs: list[RetrievedDocument],
    make: Optional[str] = None,
) -> str:
    """
    Evaluates retrieval quality with manufacturer-aware checking.
    """
    if not docs:
        return "insufficient"

    top_score = docs[0].score

    # Manufacturer check: if make is specified and NONE of the
    # retrieved docs match that make, the results are wrong docs
    if make:
        matching_make = [
            d for d in docs
            if make.lower() in d.make.lower()
            or d.make.lower() in make.lower()
        ]
        if not matching_make:
            logger.info(
                f"Quality: insufficient "
                f"(make '{make}' not found in any retrieved doc)"
            )
            return "insufficient"

    relevant_docs = [d for d in docs if d.score >= PARTIAL_SCORE]

    if top_score >= SUFFICIENT_SCORE and len(relevant_docs) >= MIN_RELEVANT_DOCS:
        return "sufficient"
    elif top_score >= PARTIAL_SCORE:
        return "partial"
    else:
        return "insufficient"

# Common non-US or excluded brands (which is not in database) to detect in free text queries
NON_NHTSA_BRANDS = [
    "maruti", "suzuki", "tata", "mahindra", "bajaj", "hero",
    "hyundai india", "kia india", "byd", "geely", "chery",
    "great wall", "haval", "renault", "peugeot", "citroen",
    "dacia", "skoda", "seat", "lada", "proton", "perodua",
]

def _extract_make_from_query(query: str) -> Optional[str]:
    """
    Detects if a known non-NHTSA brand is mentioned in the query.
    Returns the brand name so the manufacturer check can run
    even when the user hasn't filled in the vehicle filter.
    """
    query_lower = query.lower()
    for brand in NON_NHTSA_BRANDS:
        if brand in query_lower:
            return brand
    return None

# Main CRAG ask function

def ask(
    query: str,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K,
    conversation_history: Optional[list[dict]] = None,
) -> tuple[str, str, list[RetrievedDocument], list[WebSearchResult], str]:
    """
    Full CRAG pipeline: retrieve -> evaluate -> optionally search web -> generate.

    Args:
        query:                User's natural language question
        make:                 Optional vehicle make filter
        model:                Optional vehicle model filter
        year:                 Optional model year filter
        top_k:                Number of ChromaDB docs to retrieve
        conversation_history: Previous chat turns for multi-turn support

    Returns tuple of:
        answer       (str)                    — LLM-generated answer
        sources      (str)                    — Human-readable source summary
        internal_docs (list[RetrievedDocument]) — ChromaDB results
        web_results  (list[WebSearchResult])  — Tavily results (may be empty)
        quality      (str)                    — "sufficient"/"partial"/"insufficient"
    """
    logger.info(
        f"CRAG query: '{query[:60]}' | "
        f"Vehicle: {make} {model} {year}"
    )

    # If make not provided via filter, try to detect it from query text
    detected_make = make or _extract_make_from_query(query)

    # 1 :Retrieve from ChromaDB
    internal_docs = retrieve_all(
        query=query,
        top_k=top_k,
        make=make,
        model=model,
        year=year,
    )

    # 2: Evaluate retrieval quality
    # 'make' is used for chromaDB filtering but 'detected_make' is used for quality gate, so even user does not fill the filter,
    # the brand name extract from web search
    quality = evaluate_retrieval_quality(internal_docs, make=detected_make)

    # 3: Web search if needed
    web_results = []
    if quality in ("partial", "insufficient"):
        logger.info(
            f"Quality '{quality}' — triggering Tavily web search..."
        )
        raw_web = search_web(
            query=query,
            make=make,
            model=model,
            year=year,
        )
        web_results = filter_relevant_results(raw_web, make, model)
        logger.info(
            f"Tavily returned {len(web_results)} relevant results."
        )

    # 4: generate answer from combined context
    answer = generate_answer(
        query=query,
        documents=internal_docs,
        web_results=web_results,
        quality=quality,
        make=make,
        model=model,
        year=year,
        conversation_history=conversation_history,
    )

    # 5: build source summary for UI 
    sources = _build_source_summary(internal_docs, web_results, quality)

    return answer, sources, internal_docs, web_results, quality


# source summary

def _build_source_summary(
    internal_docs: list[RetrievedDocument],
    web_results: list[WebSearchResult],
    quality: str,
) -> str:
    """
    Builds a human-readable source summary shown below each answer.

    Examples:
        "Based on 4 NHTSA complaints and 1 official recall."
        "Based on 2 NHTSA complaints + 3 web sources."
        "No NHTSA data found. Answer based on 4 web sources."
    """
    n_complaints = sum(
        1 for d in internal_docs if d.doc_type == "complaint"
    )
    n_recalls = sum(
        1 for d in internal_docs if d.doc_type == "recall"
    )
    n_web = len(web_results)

    parts = []

    if quality == "sufficient":
        if n_complaints:
            parts.append(
                f"{n_complaints} NHTSA complaint"
                f"{'s' if n_complaints > 1 else ''}"
            )
        if n_recalls:
            parts.append(
                f"{n_recalls} official recall"
                f"{'s' if n_recalls > 1 else ''}"
            )
        summary = f"Based on {' and '.join(parts)}."

    elif quality == "partial":
        if n_complaints:
            parts.append(
                f"{n_complaints} NHTSA complaint"
                f"{'s' if n_complaints > 1 else ''}"
            )
        if n_recalls:
            parts.append(
                f"{n_recalls} official recall"
                f"{'s' if n_recalls > 1 else ''}"
            )
        internal_str = " and ".join(parts) if parts else "limited NHTSA data"
        web_str = f"{n_web} web source{'s' if n_web > 1 else ''}" if n_web else ""
        if web_str:
            summary = f"Based on {internal_str} + {web_str}."
        else:
            summary = f"Based on {internal_str}."

    else:
        # insufficient or web only
        if n_web:
            summary = (
                f"No NHTSA data found. "
                f"Answer based on {n_web} web source"
                f"{'s' if n_web > 1 else ''}."
            )
        else:
            summary = "No data found in NHTSA database or web search."

    # Add severity flags if any internal docs have crash/fire
    has_crash = any(d.is_crash for d in internal_docs)
    has_fire = any(d.is_fire for d in internal_docs)
    flags = []
    if has_crash:
        flags.append("crash reports")
    if has_fire:
        flags.append("fire reports")
    if flags:
        summary = summary.rstrip(".") + f" (includes {', '.join(flags)})."

    return summary


# Status check
def get_status() -> dict:
    """Returns current RAG system status for UI display."""
    status = get_index_status()
    status["ready"] = (
        status.get("complaints_indexed", 0) > 0
        or status.get("recalls_indexed", 0) > 0
    )
    return status