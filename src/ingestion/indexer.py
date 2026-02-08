"""
Indexer module for vector storage.

Handles uploading and managing embeddings in the Pinecone vector database.
"""

import json
import logging
import os
import sys
import time

from pinecone import Pinecone, ServerlessSpec

from src.config import settings
from src.models.schema import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


def get_or_create_index(index_name: str | None = None, dimension: int = 768):
    #get existing pinecone index or create a new one

    if index_name is None:
        index_name = settings.pinecone_index_name

    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for index to be ready...")
            time.sleep(1)

        logger.info(f"Index '{index_name}' is ready")

    return pc.Index(index_name)


def upsert_vectors(
    index,
    child_ids: list[str],
    embeddings: list[list[float]],
    metadata: list[dict],
    batch_size: int = 100,
):
    #upsert vectors to pinecone in batches

    vectors = [
        (cid, emb, meta)
        for cid, emb, meta in zip(child_ids, embeddings, metadata)
    ]

    total_batches = (len(vectors) + batch_size - 1) // batch_size

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        batch_num = i // batch_size + 1

        index.upsert(vectors=batch)
        logger.info(f"Upserted batch {batch_num}/{total_batches}")


def upsert_chunks(
    index,
    child_chunks: list[ChildChunk],
    embeddings: list[list[float]],
    batch_size: int = 100,
):
    #upsert child chunks with embeddings to pinecone

    vectors = []
    for chunk, embedding in zip(child_chunks, embeddings):
        vectors.append((
            chunk.child_id,
            embedding,
            {
                "parent_id": chunk.parent_id,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "text": chunk.text[:500],
            },
        ))

    total_batches = (len(vectors) + batch_size - 1) // batch_size

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        batch_num = i // batch_size + 1

        index.upsert(vectors=batch)
        logger.info(f"Upserted batch {batch_num}/{total_batches}")


def save_parent_chunks(
    parents: list[ParentChunk],
    output_path: str = "data/processed/parents.json",
):
    #save parent chunks to a local json file

    data = [p.model_dump() for p in parents]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(parents)} parent chunks to {output_path}")


def main():
    #run this to upload embeddings to pinecone

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from src.ingestion.chunking import load_chunks
    from src.ingestion.embedder import load_embeddings

    print("=== Step 3: Upload to Pinecone ===\n")

    #load chunks and embeddings from previous steps
    parents, children = load_chunks()
    child_ids, embeddings = load_embeddings()

    print(f"Loaded {len(children)} chunks and {len(embeddings)} embeddings\n")

    #build metadata from children
    child_map = {c.child_id: c for c in children}
    metadata = []
    for cid in child_ids:
        chunk = child_map[cid]
        metadata.append({
            "parent_id": chunk.parent_id,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "text": chunk.text[:500],
        })

    #upload to pinecone
    index = get_or_create_index()
    upsert_vectors(index, child_ids, embeddings, metadata)

    print(f"\nUploaded {len(embeddings)} vectors to Pinecone index: {settings.pinecone_index_name}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    main()
