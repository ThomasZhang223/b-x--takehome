"""
Study Guide Agent — Generates a comprehensive study guide using RAG content.

Runs after the Planner to append detailed content (definitions, formulas, questions)
to the final study plan.
"""

import logging
from typing import AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel, Field

from src.config import settings
from src.models.schema import TopicNode


class Definition(BaseModel):
    term: str = Field(description="The technical term being defined")
    definition: str = Field(description="The precise definition from the text")


class Formula(BaseModel):
    name: str = Field(description="Name of the formula or relationship")
    latex: str = Field(description="LaTeX representation of the formula")
    variable_descriptions: str = Field(description="Description of what each variable in the formula represents")


class PracticeQuestion(BaseModel):
    question: str = Field(description="The conceptual or calculation question")
    answer_key: str = Field(description="Brief explanation of the answer or hint")


class StudyGuideContent(BaseModel):
    executive_summary: str = Field(description="A high-level 2-3 sentence overview of the topic's importance")
    key_concepts: list[str] = Field(description="Detailed bulleted list of core concepts and their mechanics")
    definitions: list[Definition] = Field(description="List of terms and their precise definitions")
    formulas: list[Formula] = Field(description="List of formulas and their descriptions")
    analogies_and_mnemonics: list[str] = Field(description="Creative analogies or memory aids to help understand complex parts")
    common_pitfalls: list[str] = Field(description="Common misconceptions or errors students make with this material")
    practice_questions: list[PracticeQuestion] = Field(description="Practice questions with answer keys")


class StudyGuideAgent(BaseAgent):
    """Study Guide Agent — RAG Generation step.

    Reads from state: topics, retrieved_contexts
    Writes to: Appends to study_plan.md
    """

    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__(name="StudyGuideAgent")
        self.logger = logging.getLogger("agent.StudyGuide")
        self.client = genai.Client(api_key=settings.retrieval_gemini_api_key)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator:
        topics = [TopicNode(**t) for t in ctx.session.state.get("topics", [])]
        contexts = ctx.session.state.get("retrieved_contexts", [])
        output_path = ctx.session.state.get("output_path", "study_plan.md")

        print()
        print("=" * 60)
        print("  Study Guide Agent: Generating Detailed Content...")
        print("=" * 60)
        print()

        guide_sections = ["", "# Comprehensive Study Guide", "---", ""]

        for topic in topics:
            # Get relevant content
            topic_ctxs = [c for c in contexts if c.get("topic_id") == topic.topic_id]
            
            if not topic_ctxs:
                continue

            # Combine text for context
            full_text = "\n\n".join([c.get("parent_chunk_text", "") for c in topic_ctxs])
            if len(full_text) > 40000:
                full_text = full_text[:40000] + "...(truncated)"

            print(f"  Synthesizing rich content for: {topic.raw_name}...")

            prompt = f"""You are a world-class university professor creating an exhaustive, high-fidelity study guide for: {topic.raw_name} ({topic.course}).
            
Your task is to transform the following raw textbook excerpts into a "Golden Study Guide". 
Strictly adhere to the provided text; do not hallucinate external information, but use your pedagogical expertise to organize it perfectly.

TEXTBOOK CONTENT:
{full_text}

STUDY GUIDE REQUIREMENTS:
1. Executive Summary: Explain WHY this topic matters in the context of the course.
2. Key Concepts: Don't just list them; explain the relationship between variables or ideas.
3. Formulas: Provide the LaTeX and explain every single variable in the formula.
4. Pitfalls: Identify what usually trips students up in this specific text.
5. Analogies: Use one real world analogy to make the most abstract concept in the text concrete.

OUTPUT: Return a JSON object matching the requested schema.
"""
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": StudyGuideContent,
                    },
                )
                
                content = StudyGuideContent.model_validate_json(response.text)
                
                # Format markdown
                guide_sections.append(f"# {topic.raw_name}")
                guide_sections.append(f"> **Overview:** {content.executive_summary}\n")
                
                guide_sections.append("### Key Concepts")
                for concept in content.key_concepts:
                    guide_sections.append(f"- {concept}")
                guide_sections.append("")
                
                if content.definitions:
                    guide_sections.append("### Terminology")
                    for d in content.definitions:
                        guide_sections.append(f"- **{d.term}**: {d.definition}")
                    guide_sections.append("")

                if content.formulas:
                    guide_sections.append("### Mathematical Framework")
                    for f in content.formulas:
                        guide_sections.append(f"**{f.name}:**")
                        guide_sections.append(f"$${f.latex}$$")
                        guide_sections.append(f"*{f.variable_descriptions}*\n")
                    guide_sections.append("")

                if content.analogies_and_mnemonics:
                    guide_sections.append("### Mental Models & Analogies")
                    for a in content.analogies_and_mnemonics:
                        guide_sections.append(f"- {a}")
                    guide_sections.append("")

                if content.common_pitfalls:
                    guide_sections.append("### Common Pitfalls")
                    for p in content.common_pitfalls:
                        guide_sections.append(f"- {p}")
                    guide_sections.append("")

                if content.practice_questions:
                    guide_sections.append("### Practice Questions")
                    for i, q in enumerate(content.practice_questions, 1):
                        guide_sections.append(f"{i}. **{q.question}**")
                        guide_sections.append(f"   - *Hint/Key:* {q.answer_key}")
                
                guide_sections.append("\n---\n")
                print(f"  [OK] Detailed guide generated for {topic.raw_name}")

            except Exception as e:
                self.logger.error(f"Failed to generate guide for {topic.raw_name}: {e}")
                print(f"  [ERR] Failed synthesis for {topic.raw_name}")

        # Append to the existing file
        with open(output_path, "a") as f:
            f.write("\n".join(guide_sections))

        print()
        print(f"  Study guide appended to {output_path}")

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Study guide generation complete. Added to {output_path}")]
            )
        )