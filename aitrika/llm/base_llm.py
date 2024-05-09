from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Base LLM class.
    """

    @abstractmethod
    def query(self, query: str) -> str:
        """
        Query method for LLM.

        Args:
            query (str): Query

        Returns:
            str: Response
        """
        pass
