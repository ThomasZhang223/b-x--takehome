"""
Parser tool for extracting structured data from documents.

Parses PDFs to extract syllabi, exam topics, and textbook content using Gemini.
"""

import json
import logging
import os
import sys

import fitz
from google import genai
from pydantic import BaseModel, Field

from src.config import settings
from src.models.schema import TopicNode

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.embedding_gemini_api_key)

COURSES = {
    "PHYS 234": {
        "overview": "data/Midterm Topics/PHYS 234 - Midterm 1 Overview.pdf",
        "syllabus": "data/Syllabi/PHYS 234 - Syllabus.pdf",
    },
    "SYSD 300": {
        "overview": "data/Midterm Topics/SYSD 300 - Midterm 1 Overview.pdf",
        "syllabus": "data/Syllabi/SYSD 300 - Syllabus.pdf",
    },
    "HLTH 204": {
        "overview": "data/Midterm Topics/HLTH 204 - Midterm 1 Overview.pdf",
        "syllabus": "data/Syllabi/HLTH 204 - Syllabus.pdf",
    },
}


#response schema for structured output
class Topic(BaseModel):
    topic_name: str = Field(description="Broad, consolidated topic name covering a chapter or major concept")
    keywords: list[str] = Field(description="5-10 specific terms/formulas/concepts for validation and retrieval")


class CourseTopics(BaseModel):
    topics: list[Topic] = Field(description="List of consolidated topics for the midterm")


def extract_pdf_text(pdf_path: str) -> str:
    #extract all text from a pdf file

    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text("text"))

    doc.close()
    return "\n".join(text_parts)


def parse_course_topics(
    overview_path: str, syllabus_path: str, course_name: str
) -> list[TopicNode]:
    #extract topics from midterm overview and syllabus using gemini
    #consolidates into fewer broad topics to minimize embeddings

    overview_text = extract_pdf_text(overview_path)
    syllabus_text = extract_pdf_text(syllabus_path)

    prompt = f"""You are parsing a university course midterm overview and syllabus for: {course_name}.

IMPORTANT: Consolidate related topics into BROAD chapter-level topics. We want FEWER topics (aim for 5-10 per course), not a granular list. Each topic should cover a major concept or chapter.

For each consolidated topic:
1. Give it a broad name that covers the whole concept area
2. Include 5-10 specific keywords/terms/formulas that fall under this topic (these are used for retrieval)

Example of GOOD consolidation:
- Instead of: "Newton's First Law", "Newton's Second Law", "Newton's Third Law"
- Use: "Newton's Laws of Motion" with keywords ["first law", "second law", "third law", "inertia", "F=ma", "action-reaction"]

MIDTERM OVERVIEW:
{overview_text}

SYLLABUS:
{syllabus_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": CourseTopics,
        },
    )

    #parse using pydantic
    course_topics = CourseTopics.model_validate_json(response.text)

    topic_nodes = []
    for i, topic in enumerate(course_topics.topics):
        topic_id = f"{course_name.lower().replace(' ', '_')}_{i}"
        topic_nodes.append(TopicNode(
            topic_id=topic_id,
            course=course_name,
            raw_name=topic.topic_name,
            keywords=topic.keywords,
        ))

    return topic_nodes


def parse_all_courses() -> dict[str, list[TopicNode]]:
    #parse topics for all courses

    all_topics = {}

    for course_name, paths in COURSES.items():
        logger.info(f"Parsing topics for {course_name}...")
        topics = parse_course_topics(
            overview_path=paths["overview"],
            syllabus_path=paths["syllabus"],
            course_name=course_name,
        )
        all_topics[course_name] = topics
        logger.info(f"{course_name}: extracted {len(topics)} topics")

    return all_topics


def save_parsed_topics(
    all_topics: dict[str, list[TopicNode]],
    output_path: str = "data/artifact_store/syllabi.json",
):
    #save parsed topics to json file

    data = {
        course: [t.model_dump() for t in topics]
        for course, topics in all_topics.items()
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    for course, topics in all_topics.items():
        logger.info(f"{course}: extracted {len(topics)} topics")

    logger.info(f"Saved topics to {output_path}")


def main():
    #run this to parse all syllabi and save topics

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=== Parsing Syllabi and Midterm Overviews ===\n")

    all_topics = parse_all_courses()

    for course, topics in all_topics.items():
        print(f"{course}: {len(topics)} consolidated topics")

    save_parsed_topics(all_topics)
    print(f"\nTopics saved to data/artifact_store/syllabi.json")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    main()
