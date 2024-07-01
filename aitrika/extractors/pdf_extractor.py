import re
from typing import Tuple
from PyPDF2 import PdfReader
from Bio import Entrez
from aitrika.utils.load_spacy_model import load_spacy_model
from aitrika.config import ENTREZ_EMAIL


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_title_and_authors(self) -> Tuple[str, str]:
        with open(self.pdf_path, "rb") as f:
            reader = PdfReader(f)
            first_page = reader.pages[0].extract_text()
            lines = first_page.split("\n")
            pre_header = [line.strip() for line in lines if line.strip()]
            original_header = pre_header[:]
            pre_header = [re.sub(r"\d+", "", s) for s in pre_header]

            authors = self._detect_authors(pre_header)
            title = self._detect_title(pre_header)

            original_title = original_header[pre_header.index(title)]
            original_authors = original_header[pre_header.index(authors)]

            title = re.sub(r"\b\d+\b", "", original_title).strip()

            authors = re.sub(r"\d+", "", original_authors).strip().split(",")
            authors = [author.replace("*", "") for author in authors]
            authors = ", ".join(authors)
            authors = re.sub(r"\b(and)\b", "", authors)
            authors = re.sub(r",\s+", ", ", authors).strip()

        return title, authors

    def _detect_authors(self, text: list) -> str:
        nlp = load_spacy_model()
        for s in text:
            doc = nlp(s)
            names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if names:
                return s
        return None

    def _detect_title(self, text: list) -> str:
        author_string = self._detect_authors(text)
        abstract_string = next(
            (s for s in text if s.lower().startswith("abstract")), None
        )
        special_strings = [s for s in text if s.startswith(("[", "{", "("))]
        title_strings = [
            s
            for s in text
            if s not in [author_string, abstract_string] + special_strings
        ]
        return title_strings[0] if title_strings else None

    def retrieve_pubmed_id(self, title: str, authors: str) -> str:
        query = f"({title}) AND ({authors})[Author]"
        Entrez.email = ENTREZ_EMAIL
        handle = Entrez.esearch(
            db="pubmed", rettype="medline", retmode="text", term=query
        )
        record = Entrez.read(handle)
        id_paper = record["IdList"][0]
        handle = Entrez.efetch(
            db="pubmed", id=id_paper, rettype="medline", retmode="text"
        )
        records = Entrez.read(handle)
        return records["PMID"]

    def extract_full_text(self) -> str:
        with open(self.pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
