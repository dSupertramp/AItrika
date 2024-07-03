"""
Default configs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ENTREZ_EMAIL = "mail@mail.com"


@dataclass(frozen=True)
class LLMConfig:
    DEFAULT_EMBEDDINGS = "allenai/specter2_base"
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 80
    CONTEXT_WINDOW = 2048
    NUM_OUTPUT = 256
