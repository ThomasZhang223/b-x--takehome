"""
Configuration settings for the study planner.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_gemini_api_key: str
    retrieval_gemini_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
