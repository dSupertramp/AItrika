from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def query_model(self, query: str) -> str:
        pass
