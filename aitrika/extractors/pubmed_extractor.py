from typing import Dict, List, Union
import pandas as pd
from Bio import Entrez, Medline
import requests
import json
from io import StringIO
from aitrika.config.config import Config


class PubMedExtractor:
    config = Config()

    def __init__(self, pubmed_id: str):
        self.pubmed_id = pubmed_id
        self.record = None
        self.data = None

    def fetch_paper_knowledge(self):
        Entrez.email = self.config.ENTREZ_EMAIL
        handle = Entrez.efetch(
            db="pubmed", id=self.pubmed_id, rettype="medline", retmode="text"
        )
        self.record = Medline.read(handle)

    def fetch_data_knowledge(self):
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
                    "identifier": annotation["infons"].get("identifier"),
                    "text": annotation.get("text"),
                    "type": annotation["infons"].get("type"),
                    "database": annotation["infons"].get("database"),
                    "normalized_id": annotation["infons"].get("normalized_id"),
                    "name": annotation["infons"].get("name"),
                    "biotype": annotation["infons"].get("biotype"),
                }
                new_annotation = {
                    k: v for k, v in new_annotation.items() if v is not None
                }
                if new_annotation:
                    informations.append(new_annotation)
            data = [dict(t) for t in {tuple(d.items()) for d in informations}]
            data = json.dumps(data)
        self.data = data

    def extract_pubmed_id(self) -> str:
        return self.record.get("PMID", "")

    def extract_title(self) -> str:
        return self.record.get("TI", "")

    def extract_abstract(self) -> str:
        return self.record.get("AB", "")

    def extract_other_abstract(self) -> str:
        return self.record.get("OAB", "")

    def extract_genes(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Gene"].drop_duplicates()
        return df if dataframe else df.to_json()

    def extract_diseases(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Disease"].drop_duplicates()
        return df if dataframe else df.to_json()

    def extract_species(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Species"].drop_duplicates()
        return df if dataframe else df.to_json()

    def extract_chemicals(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Chemical"].drop_duplicates()
        return df if dataframe else df.to_json()

    def extract_mutations(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        df = pd.read_json(StringIO(self.data))
        df = df[df["type"] == "Mutation"].drop_duplicates()
        return df if dataframe else df.to_json()

    def extract_associations(
        self, dataframe: bool = False
    ) -> Union[pd.DataFrame, List[Dict]]:
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
        return pd.DataFrame(associations) if dataframe else associations

    def extract_authors(self) -> str:
        raw_authors = self._extract_full_response()["PubTator3"][0]["authors"]
        return ", ".join(raw_authors)

    def extract_journal(self) -> str:
        return self._extract_full_response()["PubTator3"][0]["journal"]

    def extract_full_text(self) -> str:
        pmc_id = self.record.get("PMC", "")
        if pmc_id != "":
            handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="text")
            full_text = handle.read()
            return full_text
        else:
            return ""

    def _extract_full_response(self) -> dict:
        url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={self.pubmed_id}&full=true"
        return requests.get(url).json()
