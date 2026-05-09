"""
embeddings.py: handles embedding complaints and recalls into ChromaDB.

what it does:
- Connect to Supabase and fetch complaint/recall text
- Convert text to vectors using sentence-transformers
- Store vectors persistently in ChromaDB

have to run once to build the index:
    python -m src.rag.embeddings

After that, the index is loaded from disk on every startup.
"""

import os
import logging
from typing import Optional

import psycopg2
import psycopg2.extras
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import zipfile
import requests
import tarfile
load_dotenv()
logger = logging.getLogger(__name__)

# Config
DATABASE_URL = os.getenv("DATABASE_URL")

# Where ChromaDB stores its persistent index on disk
# On Streamlit Cloud this maps to a writable directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(base_dir, "data", "chroma_db"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_CHROMA_URL = os.getenv("HUGGINGFACE_CHROMA_URL")
# ChromaDB collection names
COMPLAINTS_COLLECTION = "complaints"
RECALLS_COLLECTION = "recalls"

# records to embed in one batch (memory efficiency)
BATCH_SIZE = 160

# Singleton clients — initialised once, reused across calls

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None

def get_embedding_model() -> SentenceTransformer:
    """
    Returns the sentence-transformer model, loading it once.
    First call takes around 5 seconds to load from disk/HuggingFace.
    Subsequent calls return the cached instance instantly.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedding_model

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client.
    Data is saved to CHROMA_PATH so it survives server restarts.
    """
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info(f"ChromaDB connected at: {CHROMA_PATH}")
    return _chroma_client

# Database helpers
def get_db_connection():
    """Returns a psycopg2 connection to Supabase."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not set in .env")
    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor
    )

# Core embedding functions

def _embed_and_upsert(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    """
    Embeds a batch of texts and upserts them into a ChromaDB collection.
    Uses upsert (not add) so re-running doesn't create duplicates.
    existing documents are updated, new ones are inserted.
    """
    if not texts:
        return

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

def build_complaints_index(force_rebuild: bool = False) -> int:
    """
    Fetches all complaints from Supabase and embeds them into ChromaDB.
    Reconnects to Supabase on every batch to avoid connection timeout.
    """
    client = get_chroma_client()
    model = get_embedding_model()

    if force_rebuild:
        try:
            client.delete_collection(COMPLAINTS_COLLECTION)
            logger.info("Deleted existing complaints collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COMPLAINTS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Check how many are already embedded so we can resume
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info(
            f"Complaints collection already has {existing_count} docs. "
            "Skipping rebuild. Use force_rebuild=True to refresh."
        )
        return existing_count

    logger.info("Building complaints index from Supabase...")

    # Get total count using a fresh short-lived connection
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) as total FROM complaints "
        "WHERE summary IS NOT NULL AND summary != ''"
    )
    total = cur.fetchone()["total"]
    conn.close()  # close immediately after count query

    logger.info(f"Embedding {total} complaints in batches of {BATCH_SIZE}")

    total_embedded = 0
    offset = 0

    while True:
        # Open a FRESH connection for every single batch
        # This prevents Supabase from closing the connection
        # after it sits idle during the embedding step
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT odi_number, make, model, model_year,
                       component, summary, crash, fire,
                       injuries, deaths, date_complained
                FROM complaints
                WHERE summary IS NOT NULL
                  AND summary != ''
                ORDER BY odi_number
                LIMIT %s OFFSET %s
                """,
                (BATCH_SIZE, offset)
            )
            rows = cur.fetchall()
        finally:
            conn.close()  # always close before embedding (embedding takes time)

        if not rows:
            break

        ids, texts, metadatas = [], [], []
        for row in rows:
            doc_id = f"complaint_{row['odi_number']}"
            component = row["component"] or ""
            summary = row["summary"] or ""
            text = f"{component}: {summary}".strip(": ")
            metadata = {
                "make": str(row["make"] or ""),
                "model": str(row["model"] or ""),
                "year": int(row["model_year"] or 0),
                "component": str(row["component"] or ""),
                "crash": bool(row["crash"]),
                "fire": bool(row["fire"]),
                "injuries": int(row["injuries"] or 0),
                "deaths": int(row["deaths"] or 0),
                "date": str(row["date_complained"] or ""),
                "type": "complaint",
            }
            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        # Embed and save to ChromaDB (connection is already closed here)
        _embed_and_upsert(collection, model, ids, texts, metadatas)
        total_embedded += len(rows)
        pct = round((total_embedded / total) * 100, 1)
        logger.info(
            f"Embedded {total_embedded}/{total} complaints ({pct}%)..."
        )
        offset += BATCH_SIZE

    final_count = collection.count()
    logger.info(f"Complaints index built: {final_count} documents.")
    return final_count

def build_recalls_index(force_rebuild: bool = False) -> int:
    """
    Fetches all recalls from Supabase and embeds them into ChromaDB.

    Returns:
        Number of documents in the collection after indexing.
    """
    client = get_chroma_client()
    model = get_embedding_model()

    if force_rebuild:
        try:
            client.delete_collection(RECALLS_COLLECTION)
            logger.info("Deleted existing recalls collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=RECALLS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info(
            f"Recalls collection already has {existing_count} docs. "
            "Skipping rebuild."
        )
        return existing_count

    logger.info("Building recalls index from Supabase...")
    conn = get_db_connection()
    total_embedded = 0

    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) as total FROM recalls")
        total = cur.fetchone()["total"]
        logger.info(f"Embedding {total} recalls...")

        offset = 0
        while True:
            cur.execute(
                """
                SELECT campaign_number, manufacturer, make, model,
                       model_year, component, summary,
                       consequence, remedy, recall_date, park_it
                FROM recalls
                WHERE summary IS NOT NULL
                  AND summary != ''
                ORDER BY campaign_number
                LIMIT %s OFFSET %s
                """,
                (BATCH_SIZE, offset)
            )
            rows = cur.fetchall()
            if not rows:
                break

            ids, texts, metadatas = [], [], []
            for row in rows:
                doc_id = f"recall_{row['campaign_number']}"

                # Richer text for recalls — include consequence and remedy
                parts = [
                    row["component"] or "",
                    row["summary"] or "",
                    row["consequence"] or "",
                    row["remedy"] or "",
                ]
                text = " ".join(p for p in parts if p).strip()

                metadata = {
                    "make": str(row["make"] or ""),
                    "model": str(row["model"] or ""),
                    "year": int(row["model_year"] or 0),
                    "component": str(row["component"] or ""),
                    "manufacturer": str(row["manufacturer"] or ""),
                    "recall_date": str(row["recall_date"] or ""),
                    "park_it": bool(row["park_it"]),
                    "campaign_number": str(row["campaign_number"] or ""),
                    "type": "recall",
                }

                ids.append(doc_id)
                texts.append(text)
                metadatas.append(metadata)

            _embed_and_upsert(collection, model, ids, texts, metadatas)
            total_embedded += len(rows)
            offset += BATCH_SIZE

    finally:
        conn.close()

    final_count = collection.count()
    logger.info(f"Recalls index built: {final_count} documents.")
    return final_count

def build_full_index(force_rebuild: bool = False) -> dict:
    """
    Builds both complaints and recalls indexes.
    Call this once to set up ChromaDB, then it loads from disk on restart.

    Returns:
        Dict with complaint and recall counts.
    """
    complaints_count = build_complaints_index(force_rebuild=force_rebuild)
    recalls_count = build_recalls_index(force_rebuild=force_rebuild)
    return {
        "complaints": complaints_count,
        "recalls": recalls_count,
    }

def _download_index_from_hf():
    if not HF_CHROMA_URL:
        logger.error("HUGGINGFACE_CHROMA_URL not set")
        return False
    try:
        # Use absolute path based on file location
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        tar_path = os.path.join(data_dir, "chroma_db.tar.gz")

        logger.info(f"Downloading ChromaDB from: {HF_CHROMA_URL}")
        r = requests.get(HF_CHROMA_URL, timeout=600)
        r.raise_for_status()

        with open(tar_path, "wb") as f:
            f.write(r.content)
        logger.info(f"Downloaded: {os.path.getsize(tar_path)} bytes")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        os.remove(tar_path)

        logger.info(f"Extracted to: {data_dir}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {type(e).__name__}: {e}")
        return False
    
def get_index_status() -> dict:
    try:
        client = get_chroma_client()
        complaints_col = client.get_or_create_collection(COMPLAINTS_COLLECTION)
        
        if complaints_col.count() == 0:
            # Try downloading pre-built index from HuggingFace
            downloaded = _download_index_from_hf()
            if downloaded:
                # Reset client to pick up new files
                global _chroma_client
                _chroma_client = None
                client = get_chroma_client()
                complaints_col = client.get_or_create_collection(
                    COMPLAINTS_COLLECTION
                )

        recalls_col = client.get_or_create_collection(RECALLS_COLLECTION)
        return {
            "complaints_indexed": int(complaints_col.count()),
            "recalls_indexed": int(recalls_col.count()),
            "chroma_path": CHROMA_PATH,
            "embedding_model": EMBEDDING_MODEL,
        }
    except Exception as e:
        logger.error(f"Could not get index status: {e}")
        return {"complaints_indexed": 0, "recalls_indexed": 0, "error": str(e)}

# CLI entrypoint to run directly to build the index
# python -m src.rag.embeddings --force to force rebuild if it gets disconnected or damaged

import sys
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    force = "--force" in sys.argv
    if force:
        logger.info("Force rebuild requested — deleting existing indexes...")

    result = build_full_index(force_rebuild=force)
    print(f"\nIndex build complete:")
    print(f"  Complaints: {result['complaints']} documents")
    print(f"  Recalls:    {result['recalls']} documents")
    print(f"\nto rebuild from scratch: python -m src.rag.embeddings --force")