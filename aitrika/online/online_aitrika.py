from typing import Optional
from aitrika.base.aitrika_base import AItrikaBase
from aitrika.utils.loader import loader


class OnlineAItrika(AItrikaBase):
    def __init__(self, pubmed_id: str):
        super().__init__(pubmed_id)
        self._paper_knowledge()
        self._data_knowledge()

    def _paper_knowledge(self):
        self.pubmed_extractor.fetch_paper_knowledge()

    def _data_knowledge(self):
        self.pubmed_extractor.fetch_data_knowledge()

    @loader(text="Extracting full text")
    def extract_full_text(self) -> str:
        return self.pubmed_extractor.extract_full_text()
