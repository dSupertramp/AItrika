from aitrika.base.aitrika_base import AItrikaBase
from aitrika.extractors.pdf_extractor import PDFExtractor
from aitrika.utils.loader import loader


class LocalAItrika(AItrikaBase):
    def __init__(self, pdf_path: str):
        super().__init__()
        self.pdf_path = pdf_path
        self.pdf_extractor = PDFExtractor(pdf_path)
        self.title = None
        self.authors = None
        self._extract_title_and_authors()
        self._retrieve_pubmed_id()

    def _extract_title_and_authors(self):
        self.title, self.authors = self.pdf_extractor.extract_title_and_authors()

    def _retrieve_pubmed_id(self):
        self.pubmed_id = self.pdf_extractor.retrieve_pubmed_id(self.title, self.authors)
        if self.pubmed_id:
            super().__init__(self.pubmed_id)
            self._paper_knowledge()
            self._data_knowledge()

    def _paper_knowledge(self):
        if self.pubmed_extractor:
            self.pubmed_extractor.fetch_paper_knowledge()

    def _data_knowledge(self):
        if self.pubmed_extractor:
            self.pubmed_extractor.fetch_data_knowledge()

    @loader(text="Extracting full text")
    def extract_full_text(self) -> str:
        return self.pdf_extractor.extract_full_text()
