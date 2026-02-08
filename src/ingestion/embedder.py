"""
Embedder module for generating vector embeddings.

Converts text chunks into vector representations using Google's text-embedding-004 model.
"""

import json
import logging
import os
import sys
import time

from google import genai

from src.config import settings

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.retrieval_gemini_api_key)

#in-memory cache for query embeddings to avoid re-embedding during runtime
_query_embedding_cache: dict[str, list[float]] = {}


def embed_texts(texts: list[str], batch_size: int = 15) -> list[list[float]]:

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        # batching chunks but that doesn't even work for reducing gemini rate limits
        # gemini calculates on api call per chunk embedded
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(texts)} total chunks)")

        backoff = 10
        max_backoff = 60

        while True: # retry logic to wait for RPM
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                break
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Rate limited, waiting {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                else:
                    raise

        time.sleep(10)  # Wait 10s between batches to stay under 100 RPM

    return all_embeddings


def embed_query(query: str) -> list[float]:
    #embed single query for retrieval, with caching

    #check cache first
    if query in _query_embedding_cache:
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return _query_embedding_cache[query]

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    embedding = result.embeddings[0].values

    #cache the result
    _query_embedding_cache[query] = embedding
    return embedding


def get_query_cache() -> dict[str, list[float]]:
    #return the query embedding cache for persistence
    return _query_embedding_cache


def load_query_cache(cache: dict[str, list[float]]) -> None:
    #load query embedding cache from session state
    global _query_embedding_cache
    _query_embedding_cache.update(cache)


def save_embeddings(
    embeddings: list[list[float]],
    child_ids: list[str],
    output_path: str = "data/processed/embeddings.json",
):
    #save embeddings to json so we dont lose them if upload fails

    data = [
        {"child_id": cid, "embedding": emb}
        for cid, emb in zip(child_ids, embeddings)
    ]

    with open(output_path, "w") as f:
        json.dump(data, f)

    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")


def load_embeddings(
    input_path: str = "data/processed/embeddings.json",
) -> tuple[list[str], list[list[float]]]:
    #load embeddings from json for resuming pipeline

    with open(input_path) as f:
        data = json.load(f)

    child_ids = [d["child_id"] for d in data]
    embeddings = [d["embedding"] for d in data]

    return child_ids, embeddings


def main():
    #run this to embed all chunks and save locally

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from src.ingestion.chunking import load_chunks

    print("=== Step 2: Embedding Child Chunks ===\n")

    #load chunks from previous step
    parents, children = load_chunks()
    print(f"Loaded {len(children)} child chunks to embed\n")

    #embed all child texts
    texts = [c.text for c in children]
    child_ids = [c.child_id for c in children]

    embeddings = embed_texts(texts)

    print(f"\nEmbedded {len(embeddings)} chunks (768-dim vectors)")

    #save embeddings locally
    save_embeddings(embeddings, child_ids)
    print("Embeddings saved to data/processed/embeddings.json")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    main()
