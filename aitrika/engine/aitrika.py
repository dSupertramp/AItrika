from Bio import Entrez, Medline
import pandas as pd
import requests
import json
from io import StringIO
from PyPDF2 import PdfReader
import re
from aitrika.prompts.prompts import (
    results_prompt,
    bibliography_prompt,
    methods_prompt,
    introduction_prompt,
    acknowledgments_prompt,
    paper_results_prompt,
    effect_sizes_prompt,
    number_of_participants_prompt,
    characteristics_of_participants_prompt,
    interventions_prompt,
    outcomes_prompt,
)
from aitrika.utils.loader import loader
from aitrika.utils.load_spacy_model import load_spacy_model
from aitrika.llm.base_llm import BaseLLM


class AItrikaBase:
    """
    Base AItrika class.
    """

    def __init__(self):
        self.record = None
        self.data = None

    def _paper_knowledge(self):
        """
        Extract the knowledge of the paper (title, abstract etc.).
        """
        Entrez.email = "mail@mail.com"
        handle = Entrez.efetch(
            db="pubmed", id=self.pubmed_id, rettype="medline", retmode="text"
        )
        self.record = Medline.read(handle)

    def _data_knowledge(self):
        """
        Extract knowledge of the content of the paper (genes, diseases etc.).
        """
        url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={self.pubmed_id}&full=true"
        response = requests.get(url).json()
        if response == {"detail": "Could not retrieve publications"}:
            raise ValueError("Resource not found.")
        annotations, informations = [], []
        for item in response["PubTator3"]:
            for passage in item["passages"]:
                annotations.extend(passage["annotations"])
            for annotation in annotations:
                new_annotation = {
                    "identifier": annotation["infons"]["identifier"],
                    "text": annotation["text"],
                    "type": annotation["infons"]["type"],
                    "database": annotation["infons"]["database"],
                    "normalized_id": annotation["infons"]["normalized_id"],
                    "name": annotation["infons"]["name"],
                    "biotype": annotation["infons"]["biotype"],
                }
                informations.append(new_annotation)
            data = [dict(t) for t in {tuple(d.items()) for d in informations}]
            data = json.dumps(data)
        self.data = data

    def _extract_full_response(self) -> json:
        """
        Extract full response from PubTator API.

        Returns:
            json: JSON Response
        """
        url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={self.pubmed_id}&full=true"
        response = requests.get(url).json()
        return response

    @loader(text="Extracting PubMed ID")
    def extract_pubmed_id(self) -> str:
        """
        Extract PubMed ID.

        Returns:
            str: PubMed ID
        """
        return self.record.get("PMID", "")

    @loader(text="Extracting title")
    def extract_title(self) -> str:
        """
        Extract title.

        Returns:
            str: Title
        """
        return self.record.get("TI", "")

    @loader(text="Extracting abstract")
    def extract_abstract(self) -> str:
        """
        Extract abstract.

        Returns:
            str: Abstract
        """
        return self.record.get("AB", "")

    @loader(text="Extracting other abstract")
    def extract_other_abstract(self) -> str:
        """
        Extract other abstract (if available).

        Returns:
            str: Other abstract
        """
        return self.record.get("OAB", "")

    @loader(text="Extracting genes")
    def extract_genes(self, dataframe: bool = False):
        """
        Extract genes.

        Args:
            dataframe (bool, optional): Format into DataFrame. Defaults to False.

        Returns:
            pd.DataFrame | json: Genes
        """
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Gene"]
        df = df.drop_duplicates()
        if dataframe:
            return df
        else:
            return df.to_json()

    @loader(text="Extracting diseases")
    def extract_diseases(self, dataframe: bool = False):
        """
        Extract diseases.

        Args:
            dataframe (bool, optional): Format into DataFrame. Defaults to False.

        Returns:
            pd.DataFrame | json: Genes
        """
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Disease"]
        df = df.drop_duplicates()
        if dataframe:
            return df
        else:
            return df.to_json()

    @loader(text="Extracting species")
    def extract_species(self, dataframe: bool = False):
        """
        Extract species.

        Args:
            dataframe (bool, optional): Format into DataFrame. Defaults to False.

        Returns:
            pd.DataFrame | json: Genes
        """
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Species"]
        df = df.drop_duplicates()
        if dataframe:
            return df
        else:
            return df.to_json()

    @loader(text="Extracting chemicals")
    def extract_chemicals(self, dataframe: bool = False):
        """
        Extract chemicals.

        Args:
            dataframe (bool, optional): Format into DataFrame. Defaults to False.

        Returns:
            pd.DataFrame | json: Genes
        """
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Chemical"]
        df = df.drop_duplicates()
        if dataframe:
            return df
        else:
            return df.to_json()

    @loader(text="Extracting mutations")
    def extract_mutations(self, dataframe: bool = False):
        """
        Extract mutations.

        Args:
            dataframe (bool, optional): Format into DataFrame. Defaults to False.

        Returns:
            pd.DataFrame | json: Genes
        """
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Mutation"]
        df = df.drop_duplicates()
        if dataframe:
            return df
        else:
            return df.to_json()

    @loader(text="Extracting associations between genes and diseases")
    def extract_associations(self, dataframe: bool = False):
        """
        Extract associations between genes and diseases.

        Args:
            dataframe (bool, optional): Format into DataFrame. Defaults to False.

        Returns:
            pd.DataFrame | json: Genes
        """
        relations, associations = [], []
        data = self._extract_full_response()
        for item in data["PubTator3"]:
            relations.extend(item["relations_display"])
        for item in relations:
            name = item["name"]
            parts = name.split("|")
            disease = parts[1].replace("@DISEASE_", "").replace("_", " ")
            gene = parts[2].replace("@GENE_", "")
            associations.append({"gene": gene, "disease": disease})
        if dataframe:
            return pd.DataFrame(associations)
        else:
            return associations

    @loader(text="Extracting authors")
    def extract_authors(self):
        """
        Extract authors.

        Returns:
            str: Authors
        """
        raw_authors = self._extract_full_response()["PubTator3"][0]["authors"]
        return ", ".join(raw_authors)

    @loader(text="Extracting journal")
    def extract_journal(self):
        """
        Extract journal.

        Returns:
            str: Journal
        """
        return self._extract_full_response()["PubTator3"][0]["journal"]

    @loader(text="Extracting results")
    def extract_results(self, llm: BaseLLM) -> str:
        """
        Extract results.

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=results_prompt)

    @loader(text="Extracting bibliography")
    def extract_bibliography(self, llm: BaseLLM) -> str:
        """
        Extract bibliography.

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=bibliography_prompt)

    @loader(text="Extracting methods")
    def extract_methods(self, llm: BaseLLM) -> str:
        """
        Extract methods.

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=methods_prompt)

    @loader(text="Extracting introduction")
    def extract_introduction(self, llm: BaseLLM) -> str:
        """
        Extract introduction.

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=introduction_prompt)

    @loader(text="Extracting acknowledgements")
    def extract_acknowledgements(self, llm: BaseLLM) -> str:
        """
        Extract acknowledgements.

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=acknowledgments_prompt)

    @loader(text="Extracting paper results")
    def extract_paper_results(self, llm: BaseLLM) -> str:
        """
        Extract paper results.

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=paper_results_prompt)

    @loader(text="Extracting effect sizes")
    def extract_effect_sizes(self, llm: BaseLLM) -> str:
        """
        Extract effect sizes.

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=effect_sizes_prompt)

    @loader(text="Extracting number of participants")
    def extract_number_of_participants(self, llm: BaseLLM) -> str:
        """
        Extract number_of_participants

        Args:
            llm (BaseLLM): Provided LLM

        Returns:
            str: Results
        """
        return llm.query(query=number_of_participants_prompt)

    @loader(text="Extracting characteristics of participants")
    def extract_characteristics_of_participants(self, llm: BaseLLM) -> str:
        """
        Extract characteristics of participants.

        Args:
            llm (BaseLLM): LLM

        Returns:
            str: Characteristics of participants separated by semicolon
        """
        return llm.query(query=characteristics_of_participants_prompt)

    @loader(text="Extracting interventions")
    def get_interventions(self, llm: BaseLLM) -> str:
        """
        Extract interventions.

        Args:
            llm (BaseLLM): LLM

        Returns:
            str: Interventions separated by semicolon
        """
        return llm.query(query=interventions_prompt)

    @loader(text="Extracting outcomes")
    def get_outcomes(self, llm: BaseLLM) -> str:
        """
        Extract outcomes.

        Args:
            llm (BaseLLM): LLM

        Returns:
            str: Outcomes separated by semicolon
        """
        return llm.query(query=outcomes_prompt)


class OnlineAItrika(AItrikaBase):
    """
    AItrika engine for online search.

    Args:
        AItrikaBase (): Base AItrika
    """

    def __init__(self, pubmed_id: str) -> None:
        super().__init__()
        self.pubmed_id = pubmed_id
        self._paper_knowledge()
        self._data_knowledge()

    @loader(text="Extracting full text")
    def extract_full_text(self) -> str:
        """
        Extract full text (if available):

        Returns:
            str: Full text
        """
        pmc_id = self.record.get("PMC", "")
        if pmc_id != "":
            handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="text")
            full_text = handle.read()
            return full_text
        else:
            return ""


class LocalAItrika(AItrikaBase):
    """
    Local AItrika engine for local search.

    Args:
        AItrikaBase (): Base AItrika
    """

    def __init__(self, pdf_path: str) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.title = None
        self.authors = None
        self.pubmed_id = None
        self._extract_title_and_authors()
        try:
            self._retrieve_pubmed_id()
            self._paper_knowledge()
            self._data_knowledge()
        except Exception:
            pass

    def _extract_title_and_authors(self):
        """
        Extract title and authors from PDF.
        """

        def _detect_authors(text: str) -> str:
            """
            Detect authors inside the text.

            Args:
                text (str): Input text

            Returns:
                str: Authors
            """
            nlp = load_spacy_model()
            for s in text:
                doc = nlp(s)
                names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                if names:
                    return s
            return None

        def _detect_title(text: str) -> str:
            """
            Detect title inside the text.

            Args:
                text (str): Input text

            Returns:
                str: Title
            """
            author_string = _detect_authors(text)
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

        with open(self.pdf_path, "rb") as f:
            reader = PdfReader(f)
            first_page = reader.pages[0].extract_text()
            lines = first_page.split("\n")
            pre_header = [line.strip() for line in lines if line.strip()]
            original_header = pre_header[:]  ## Copy of pre_header
            pre_header = [re.sub(r"\d+", "", s) for s in pre_header]  ## Remove numbers

            # Perform author and title detection on pre_header
            authors = _detect_authors(pre_header)
            title = _detect_title(pre_header)

            # Extract the actual title and authors from the original list
            original_title = original_header[pre_header.index(title)]
            original_authors = original_header[pre_header.index(authors)]

            ## Title
            title = re.sub(r"\b\d+\b", "", original_title)  ## Remove numbers and strip
            title = title.strip()

            ## Authors
            authors = (
                re.sub(r"\d+", "", original_authors).strip().split(",")
            )  ## Remove numbers
            authors = [author.replace("*", "") for author in authors]  ## Remove *
            authors = ", ".join(authors)
            authors = re.sub(r"\b(and)\b", "", authors)  ## Remove 'and word'
            authors = re.sub(r",\s+", ", ", authors)  ## Remove extra spaces after 'and'
            authors = authors.strip()
            self.title = title
            self.authors = authors

    def _retrieve_pubmed_id(self) -> int:
        """
        Retrieve PubMed ID based on title and authors.

        Returns:
            int: PubMed ID
        """
        query = f"({self.title}) AND ({self.authors})[Author]"
        Entrez.email = "mail@mail.com"
        handle = Entrez.esearch(
            db="pubmed", rettype="medline", retmode="text", term=query
        )
        record = Entrez.read(handle)
        id_paper = record["IdList"][0]
        handle = Entrez.efetch(
            db="pubmed", id=id_paper, rettype="medline", retmode="text"
        )
        records = Medline.read(handle)
        self.pubmed_id = records["PMID"]

    @loader(text="Extracting full text")
    def extract_full_text(self) -> str:
        """
        Extract full text parsed from PDF.

        Returns:
            str: Full text
        """
        with open(self.pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
