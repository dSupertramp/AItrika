"""
Default config for LLMs.
"""

DEFAULT_EMBEDDINGS: str = "allenai/specter2_base"
CHUNK_SIZE: int = 1024  # Size of each chunk of text
CHUNK_OVERLAP: int = 80  # Number of token that overlap between consecutive chunks
CONTEXT_WINDOW: int = (
    2048  # Maximum number of tokens that the model can consider at one time
)
NUM_OUTPUT: int = 256  # Number of tokens that the model generates in one forward pass
