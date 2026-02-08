"""
Main entry point for the AI Exam Study Planner.

Multi-Agent RAG Pipeline using Google ADK.
"""

import asyncio
import logging
import os

# IMPORTANT: Set this BEFORE any google imports
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

from google.adk.agents import LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.agents.orchestrator import OrchestratorAgent
from src.agents.estimator import EstimatorAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.validator import ValidatorAgent
from src.config import settings

#set API key for ADK/Gemini
os.environ["GOOGLE_API_KEY"] = settings.retrieval_gemini_api_key

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

#suppress noisy loggers
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─── Agent Definitions ───

orchestrator_agent = OrchestratorAgent()
researcher_agent = ResearcherAgent()
validator_agent = ValidatorAgent()
estimator_agent = EstimatorAgent()
planner_agent = PlannerAgent()

#generator-critic loop: Researcher generates, Validator critiques
#LoopAgent runs sub_agents sequentially, repeats until escalate=True or max_iterations
research_validation_loop = LoopAgent(
    name="ResearchValidationLoop",
    sub_agents=[researcher_agent, validator_agent],
    max_iterations=2,  #at most 2 passes (first pass + one retry)
)

#main sequential pipeline
#follows the pattern: Config (HITL1) -> Research+Validate (Generator-Critic) -> Estimate (HITL2) -> Plan
study_planner_pipeline = SequentialAgent(
    name="StudyPlannerPipeline",
    sub_agents=[
        orchestrator_agent,        #HITL 1: User configuration
        research_validation_loop,  #Generator-Critic: Retrieve & validate
        estimator_agent,           #HITL 2: Time estimation & user review
        planner_agent,             #Output: Generate study_plan.md
    ],
    description="Multi-agent study planner: sequential pipeline with generator-critic loop and human-in-the-loop checkpoints.",
)

# ─── Constants ───
APP_NAME = "exam_study_planner"
USER_ID = "student_1"
SESSION_ID = "session_001"


async def main():
    print()
    print("=" * 60)
    print("  b(x) Theory -- AI Exam Study Planner")
    print("  Multi-Agent RAG Pipeline (Google ADK)")
    print("=" * 60)
    print()

    #verify prerequisite data
    if not os.path.exists("data/artifact_store/syllabi.json"):
        print("  ERROR: Run parser first: python -m src.tools.parser")
        return
    if not os.path.exists("data/processed/parents.json"):
        print("  ERROR: Run ingestion first:")
        print("    python -m src.ingestion.chunking")
        print("    python -m src.ingestion.embedder")
        print("    python -m src.ingestion.indexer")
        return

    #setup ADK session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    #create runner with root agent
    runner = Runner(
        agent=study_planner_pipeline,
        app_name=APP_NAME,
        session_service=session_service,
    )

    #run the pipeline
    #send an initial message to kick off the sequential pipeline
    initial_message = types.Content(
        role="user",
        parts=[types.Part(text="Generate my exam study plan.")],
    )

    print("  Starting pipeline...")
    print()

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=initial_message,
    ):
        #log agent events
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    logging.getLogger("pipeline").info(f"[{event.author}] {part.text}")

    #done
    print()
    print("=" * 60)
    print("  Study plan generated successfully!")
    output = session.state.get("output_path", "study_plan.md")
    print(f"  Output: {output}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
