from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def query_model(self, query: str) -> str:
        pass
