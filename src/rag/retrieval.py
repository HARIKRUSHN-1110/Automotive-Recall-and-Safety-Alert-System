"""
retrieval.py: Semantic search over the ChromaDB complaint and recall indexes.

what this file does:
Given a user query, this module:
1. Embeds the query using the same model used during indexing
2. Searches ChromaDB for the most similar documents
3. Optionally filters by make/model/year
4. Returns ranked documents with their similarity scores
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .embeddings import (
    get_chroma_client,
    get_embedding_model,
    COMPLAINTS_COLLECTION,
    RECALLS_COLLECTION,
)

logger = logging.getLogger(__name__)

# Default number of documents to retrieve per query
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Result dataclass

@dataclass
class RetrievedDocument:
    """
    A single document returned by semantic search.

    text:       The original complaint or recall text
    metadata:   All stored fields (make, model, year, crash, etc.)
    score:      Cosine similarity score — higher = more relevant
                Range: 0.0 (unrelated) to 1.0 (identical)
    doc_type:   "complaint" or "recall"
    """
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    doc_type: str = "complaint"

    @property
    def make(self) -> str:
        return self.metadata.get("make", "")

    @property
    def model(self) -> str:
        return self.metadata.get("model", "")

    @property
    def year(self) -> int:
        return self.metadata.get("year", 0)

    @property
    def component(self) -> str:
        return self.metadata.get("component", "")

    @property
    def is_crash(self) -> bool:
        return bool(self.metadata.get("crash", False))

    @property
    def is_fire(self) -> bool:
        return bool(self.metadata.get("fire", False))

    def to_context_string(self) -> str:
        """
        formats the document as a readable string for the LLM prompt.
        The LLM reads this text to generate its answer.
        """
        if self.doc_type == "recall":
            return (
                f"[OFFICIAL RECALL] {self.make} {self.model} {self.year}\n"
                f"Campaign: {self.metadata.get('campaign_number', 'N/A')}\n"
                f"Date: {self.metadata.get('recall_date', 'N/A')}\n"
                f"Park it: {'YES' if self.metadata.get('park_it') else 'No'}\n"
                f"Details: {self.text}"
            )
        else:
            flags = []
            if self.is_crash:
                flags.append("CRASH INVOLVED")
            if self.is_fire:
                flags.append("FIRE INVOLVED")
            injuries = self.metadata.get("injuries", 0)
            deaths = self.metadata.get("deaths", 0)
            if injuries:
                flags.append(f"{injuries} injuries")
            if deaths:
                flags.append(f"{deaths} deaths")

            flag_str = f" [{', '.join(flags)}]" if flags else ""
            return (
                f"[COMPLAINT]{flag_str} {self.make} {self.model} {self.year}\n"
                f"Component: {self.component}\n"
                f"Date: {self.metadata.get('date', 'N/A')}\n"
                f"Report: {self.text}"
            )


# Core retrieval functions

def _build_where_filter(
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
) -> Optional[dict]:
    """
    Builds a ChromaDB metadata filter for make/model/year.
    ChromaDB filter syntax uses $and/$eq operators.
    Returns None if no filters are specified (search all documents).
    """
    conditions = []
    if make:
        conditions.append({"make": {"$eq": make.upper()}})
    if model:
        conditions.append({"model": {"$eq": model}})
    if year:
        conditions.append({"year": {"$eq": year}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_complaints(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
) -> list[RetrievedDocument]:
    """
    Search across the complaints collection for documents relevant to the query.

    Args:
        query:  Natural language question from the user
        top_k:  Number of results to return (default 5, max 20)
        make:   Optional filter by vehicle make (e.g. "BMW")
        model:  Optional filter by vehicle model (e.g. "3 Series")
        year:   Optional filter by model year (e.g. 2020)

    Returns:
        List of RetrievedDocument sorted by relevance score
    """
    top_k = min(top_k, MAX_TOP_K)

    try:
        client = get_chroma_client()
        embedding_model = get_embedding_model()
        collection = client.get_collection(COMPLAINTS_COLLECTION)

        if collection.count() == 0:
            logger.warning(
                "Complaints collection is empty. "
                "Run embeddings first to build the index."
            )
            return []

        # Embed the user query using the same model used during indexing
        query_embedding = embedding_model.encode(query).tolist()

        # Build optional metadata filter
        where = _build_where_filter(make, model, year)

        # Query ChromaDB
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        # Parse results into RetrievedDocument objects
        documents = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score (1=identical, 0=unrelated)
            score = round(1 - (distance / 2), 4)
            documents.append(RetrievedDocument(
                text=text,
                metadata=metadata,
                score=score,
                doc_type="complaint",
            ))

        logger.info(
            f"Retrieved {len(documents)} complaints for query: '{query[:50]}'"
        )
        return documents

    except Exception as e:
        logger.error(f"Complaint retrieval failed: {e}")
        return []


def retrieve_recalls(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
) -> list[RetrievedDocument]:
    """
    Searches the recalls collection for documents relevant to the query.
    Same interface as retrieve_complaints but queries the recalls index.
    """
    top_k = min(top_k, MAX_TOP_K)

    try:
        client = get_chroma_client()
        embedding_model = get_embedding_model()
        collection = client.get_collection(RECALLS_COLLECTION)

        if collection.count() == 0:
            logger.warning("Recalls collection is empty.")
            return []

        query_embedding = embedding_model.encode(query).tolist()

        where = _build_where_filter(make, model, year)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        documents = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = round(1 - (distance / 2), 4)
            documents.append(RetrievedDocument(
                text=text,
                metadata=metadata,
                score=score,
                doc_type="recall",
            ))

        logger.info(
            f"Retrieved {len(documents)} recalls for query: '{query[:50]}'"
        )
        return documents

    except Exception as e:
        logger.error(f"Recall retrieval failed: {e}")
        return []


def retrieve_all(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
) -> list[RetrievedDocument]:
    """
    Retrieves from both complaints AND recalls collections,
    then merges and re-ranks by score.

    This gives the LLM a high and valuable context mixing complaint patterns
    with official recall information and
    returns top_k documents total from the merged total results.
    """
    complaints = retrieve_complaints(
        query, top_k=top_k, make=make, model=model, year=year
    )
    recalls = retrieve_recalls(
        query, top_k=top_k, make=make, model=model, year=year
    )

    # Merge and sort by relevance score (descending)
    all_docs = complaints + recalls
    all_docs.sort(key=lambda d: d.score, reverse=True)

    # Return top_k from the merged list
    top_docs = all_docs[:top_k]

    logger.info(
        f"Merged retrieval: {len(complaints)} complaints + "
        f"{len(recalls)} recalls → top {len(top_docs)} returned"
    )
    return top_docs