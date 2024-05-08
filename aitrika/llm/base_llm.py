from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def query(self, query: str) -> str:
        pass
