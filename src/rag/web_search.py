"""
web_search.py: Tavily web search for CRAG.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Max results per search
MAX_WEB_RESULTS = 5

# Singleton client
_tavily_client: Optional[TavilyClient] = None

def get_tavily_client() -> TavilyClient:
    """Returns Tavily client, initialising once."""
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY not set. "
                "Get a free key at https://app.tavily.com and add to .env"
            )
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client

# Result dataclass
@dataclass
class WebSearchResult:
    """A single result from Tavily web search."""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source: str = "web"

    def to_context_string(self) -> str:
        """Formats for LLM prompt context."""
        return (
            f"[WEB SOURCE] {self.title}\n"
            f"URL: {self.url}\n"
            f"Content: {self.snippet}"
        )

# Query builder

def _build_search_query(
    query: str,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """
    Builds a focused automotive safety search query.
    Combines vehicle context with the user query and safety keywords
    to get relevant results rather than general pages.
    """
    parts = []

    if make:
        parts.append(make)
    if model:
        parts.append(model)
    if year:
        parts.append(str(year))

    parts.append(query)

    # Add safety context if not already in query
    if not any(
        kw in query.lower()
        for kw in ["recall", "safety", "complaint", "defect"]
    ):
        parts.append("safety recall complaints")

    return " ".join(parts)

# main search function

def search_web(
    query: str,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    max_results: int = MAX_WEB_RESULTS,
) -> list[WebSearchResult]:
    """
    Searches the web via Tavily for automotive safety information.
    Tavily's search_depth="advanced" extracts clean text from pages

    Args:
        query:       User's question
        make:        Vehicle manufacturer (e.g. "Maruti Suzuki")
        model:       Vehicle model 
        year:        Model year
        max_results: Max results to return (max 5 on free tier)

    Returns:
        List of WebSearchResult sorted by Tavily relevance score.
        Returns empty list on any failure.
    """
    search_query = _build_search_query(query, make, model, year)
    logger.info(f"Tavily search: '{search_query}'")

    try:
        client = get_tavily_client()

        response = client.search(
            query=search_query,
            search_depth="advanced",    # extracts full page content
            max_results=max_results,
            include_answer=False,       # generate our own answer
            include_raw_content=False,  # clean snippets only
            topic="general",
        )

        results = []
        for r in response.get("results", []):
            results.append(WebSearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=float(r.get("score", 0.0)),
                source="web",
            ))

        # Sort by Tavily's own relevance score
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Tavily returned {len(results)} results.")
        return results

    except ValueError as e:
        # API key not set
        logger.error(f"Tavily config error: {e}")
        return []
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

# Relevance filter

def filter_relevant_results(
    results: list[WebSearchResult],
    make: Optional[str] = None,
    model: Optional[str] = None,
) -> list[WebSearchResult]:
    """
    Keeps only results that mention the vehicle make or model.
    Reduces noise when results are too generic.
    Fall back to all results if filtering removes everything.
    """
    if not make and not model:
        return results

    filtered = []
    for r in results:
        text = f"{r.title} {r.snippet}".lower()
        make_match = make.lower() in text if make else True
        model_match = model.lower() in text if model else True
        if make_match or model_match:
            filtered.append(r) 

    if filtered:
        return filtered         
    else:
        return results