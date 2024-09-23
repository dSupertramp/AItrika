from typing import Optional, Dict, List, Union
import pandas as pd
from aitrika.extractors.pubmed_extractor import PubMedExtractor
from aitrika.extractors.llm_extractor import LLMExtractor
from aitrika.utils.loader import loader
from aitrika.llm.base_llm import BaseLLM


class AItrikaBase:
    def __init__(self, pubmed_id: Optional[str] = None):
        self.pubmed_id = pubmed_id
        self.pubmed_extractor = PubMedExtractor(pubmed_id) if pubmed_id else None
        self.llm_extractor = LLMExtractor()

    @loader(text="Extracting PubMed ID")
    def extract_pubmed_id(self) -> str:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_pubmed_id()

    @loader(text="Extracting title")
    def extract_title(self) -> str:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_title()

    @loader(text="Extracting abstract")
    def extract_abstract(self) -> str:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_abstract()

    @loader(text="Extracting other abstract")
    def extract_other_abstract(self) -> str:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_other_abstract()

    @loader(text="Extracting genes")
    def extract_genes(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_genes(dataframe)

    @loader(text="Extracting diseases")
    def extract_diseases(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_diseases(dataframe)

    @loader(text="Extracting species")
    def extract_species(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_species(dataframe)

    @loader(text="Extracting chemicals")
    def extract_chemicals(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_chemicals(dataframe)

    @loader(text="Extracting mutations")
    def extract_mutations(self, dataframe: bool = False) -> Union[pd.DataFrame, str]:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_mutations(dataframe)

    @loader(text="Extracting associations between genes and diseases")
    def extract_associations(
        self, dataframe: bool = False
    ) -> Union[pd.DataFrame, List[Dict]]:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_associations(dataframe)

    @loader(text="Extracting authors")
    def extract_authors(self) -> str:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_authors()

    @loader(text="Extracting journal")
    def extract_journal(self) -> str:
        if not self.pubmed_extractor:
            raise ValueError("PubMed ID not set")
        return self.pubmed_extractor.extract_journal()

    @loader(text="Extracting results")
    def extract_results(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_results(llm)

    @loader(text="Extracting bibliography")
    def extract_bibliography(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_bibliography(llm)

    @loader(text="Extracting methods")
    def extract_methods(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_methods(llm)

    @loader(text="Extracting introduction")
    def extract_introduction(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_introduction(llm)

    @loader(text="Extracting acknowledgements")
    def extract_acknowledgements(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_acknowledgements(llm)

    @loader(text="Extracting paper results")
    def extract_paper_results(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_paper_results(llm)

    @loader(text="Extracting effect sizes")
    def extract_effect_sizes(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_effect_sizes(llm)

    @loader(text="Extracting number of participants")
    def extract_number_of_participants(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_number_of_participants(llm)

    @loader(text="Extracting characteristics of participants")
    def extract_characteristics_of_participants(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_characteristics_of_participants(llm)

    @loader(text="Extracting interventions")
    def extract_interventions(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_interventions(llm)

    @loader(text="Extracting outcomes")
    def extract_outcomes(self, llm: BaseLLM) -> str:
        return self.llm_extractor.extract_outcomes(llm)
