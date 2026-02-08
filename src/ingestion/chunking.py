"""
Chunking module for document processing
Handles splitting documents into semantic chunks for embedding and indexing

Uses a parent child chunking strategy where parent chunks are larger sections
and child chunks are smaller overlapping windows for embedding
"""

import glob
import json
import logging
import os
import sys
import uuid

import fitz

from src.models.schema import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)

#page limits per course - only parse required chapters to save on embedding costs
PAGE_LIMITS = {
    "HLTH 204": 215,
    "PHYS 234": 201,
    "SYSD 300": 290,
}

TEXTBOOK_PREFIXES = {
    "PHYS 234": "David H McIntyre",
    "SYSD 300": "John D Sterman",
    "HLTH 204": "Marc M. Triola",
}

#chunk sizes - larger chunks = fewer embeddings = less api calls
PARENT_CHUNK_SIZE = 120000
CHILD_CHUNK_SIZE = 3000
CHILD_OVERLAP = 500


def extract_text_from_pdf(pdf_path: str, max_pages: int | None = None) -> list[dict]:
    #Extract text from each page of a PDF, up to max_pages

    pages = []
    filename = os.path.basename(pdf_path)

    doc = fitz.open(pdf_path)
    total_pages = min(len(doc), max_pages) if max_pages else len(doc)

    for page_num in range(total_pages):
        logger.info(f"Extracting page {page_num + 1}/{total_pages} from {filename}")
        page = doc[page_num]
        text = page.get_text("text")

        if len(text.strip()) < 50:
            continue

        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "source_file": filename,
        })

    doc.close()
    return pages


def create_parent_chunks(
    pages: list[dict], chunk_size: int = PARENT_CHUNK_SIZE
) -> list[ParentChunk]:
    #Split pages into parent chunks of approximately chunk_size characters

    if not pages:
        return []

    parents = []
    current_text = ""
    current_page = pages[0]["page_number"]
    source_file = pages[0]["source_file"]

    for page in pages:
        paragraphs = page["text"].split("\n\n") #textbooks should be semantically split already

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_text) + len(paragraph) + 2 > chunk_size and current_text:
                parents.append(ParentChunk(
                    parent_id=str(uuid.uuid4()),
                    source_file=source_file,
                    page_number=current_page,
                    text=current_text.strip(),
                ))
                current_text = paragraph
                current_page = page["page_number"]
            else:
                if current_text:
                    current_text += "\n\n" + paragraph
                else:
                    current_text = paragraph
                    current_page = page["page_number"]

    # account for the last chunk
    if current_text.strip():
        parents.append(ParentChunk(
            parent_id=str(uuid.uuid4()),
            source_file=source_file,
            page_number=current_page,
            text=current_text.strip(),
        ))

    return parents


def create_child_chunks(
    parent: ParentChunk,
    chunk_size: int = CHILD_CHUNK_SIZE,
    overlap: int = CHILD_OVERLAP,
) -> list[ChildChunk]:
    #Create overlapping child chunks from a parent chunk using sliding window

    children = []
    text = parent.text

    if len(text) <= chunk_size:
        child_id = str(uuid.uuid4())
        children.append(ChildChunk(
            child_id=child_id,
            parent_id=parent.parent_id,
            text=text,
            source_file=parent.source_file,
            page_number=parent.page_number,
        ))
        parent.child_ids.append(child_id)
        return children

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        child_id = str(uuid.uuid4())
        children.append(ChildChunk(
            child_id=child_id,
            parent_id=parent.parent_id,
            text=chunk_text,
            source_file=parent.source_file,
            page_number=parent.page_number,
        ))
        parent.child_ids.append(child_id)

        start += chunk_size - overlap
        if start + overlap >= len(text):
            break

    return children


def process_textbook(
    pdf_path: str, max_pages: int | None = None
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    #Process a textbook PDF into parent and child chunks

    filename = os.path.basename(pdf_path)

    pages = extract_text_from_pdf(pdf_path, max_pages)
    parents = create_parent_chunks(pages)

    all_children = []
    for parent in parents:
        children = create_child_chunks(parent)
        all_children.extend(children)

    logger.info(
        f"Processed {filename}: {len(parents)} parents, {len(all_children)} children"
    )

    return parents, all_children


def save_chunks(
    parents: list[ParentChunk],
    children: list[ChildChunk],
    parents_path: str = "data/processed/parents.json",
    children_path: str = "data/processed/children.json",
):
    #save chunks to json so we dont lose them if embedding fails

    with open(parents_path, "w") as f:
        json.dump([p.model_dump() for p in parents], f, indent=2)
    logger.info(f"Saved {len(parents)} parents to {parents_path}")

    with open(children_path, "w") as f:
        json.dump([c.model_dump() for c in children], f, indent=2)
    logger.info(f"Saved {len(children)} children to {children_path}")


def load_chunks(
    parents_path: str = "data/processed/parents.json",
    children_path: str = "data/processed/children.json",
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    #load chunks from json for resuming pipeline

    with open(parents_path) as f:
        parents = [ParentChunk(**p) for p in json.load(f)]

    with open(children_path) as f:
        children = [ChildChunk(**c) for c in json.load(f)]

    return parents, children


def find_textbook_paths() -> dict[str, str]:
    #Find textbook PDF paths by matching prefixes

    pdf_files = glob.glob("data/Textbooks/*.pdf")
    textbook_paths = {}

    for course, prefix in TEXTBOOK_PREFIXES.items():
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            if filename.startswith(prefix):
                textbook_paths[course] = pdf_path
                break

    return textbook_paths


def main():
    #run this to chunk all textbooks and save locally

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=== Step 1: Chunking Textbooks ===\n")

    textbook_paths = find_textbook_paths()
    all_parents = []
    all_children = []

    for course, pdf_path in textbook_paths.items():
        max_pages = PAGE_LIMITS.get(course)
        print(f"Processing {course} (pages 1-{max_pages})...")

        parents, children = process_textbook(pdf_path, max_pages)
        all_parents.extend(parents)
        all_children.extend(children)

        print(f"  {len(parents)} parents, {len(children)} children\n")

    print(f"Total: {len(all_parents)} parents, {len(all_children)} children")

    save_chunks(all_parents, all_children)
    print("\nChunks saved to data/processed/")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    main()
