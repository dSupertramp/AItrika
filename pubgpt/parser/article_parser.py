from typing import Tuple
from Bio import Entrez, Medline
import pandas as pd
import numpy as np
import requests
import time
from xml.etree import ElementTree
from utils.utils import create_id_folder


def search_on_pubmed(query: str) -> list:
    Entrez.email = "random@example.com"
    handle = Entrez.esearch(
        db="pubmed", sort="relevance", retmax=1, retmode="text", term=query
    )
    result = Entrez.read(handle)
    ids = result["IdList"]
    handle = Entrez.efetch(
        db="pubmed", sort="relevance", retmode="text", rettype="medline", id=ids
    )
    records = Medline.parse(handle)
    return records


def parse_article(document_id: str) -> Tuple[str, str, str, str]:
    create_id_folder(document_id=document_id)
    for record in search_on_pubmed(query=document_id):
        title = record.get("TI", "")
        abstract = record.get("AB", "")
        pubmed_id = record.get("PMID", "")
    document = title + " " + abstract
    with open(f"output/{document_id}/document.txt", "w") as f:
        f.write(document)
    return pubmed_id, title, abstract, document


def extract_mesh_terms(document_id: str) -> pd.DataFrame:
    create_id_folder(document_id=document_id)
    for record in search_on_pubmed(query=document_id):
        mesh_terms = record.get("MH", "")
    df = pd.DataFrame(data=zip(mesh_terms), columns=["element"])
    df.to_csv(f"output/{document_id}/mesh_terms.csv", encoding="utf-8", index=False)


def extract_other_terms(document_id: str) -> pd.DataFrame:
    create_id_folder(document_id=document_id)
    for record in search_on_pubmed(query=document_id):
        other_terms = record.get("OT", "")
    df = pd.DataFrame(data=zip(other_terms), columns=["element"])
    df.to_csv(f"output/{document_id}/other_terms.csv", encoding="utf-8", index=False)


def extract_genes_and_diseases(document_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    create_id_folder(document_id=document_id)
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids={document_id}&concepts=gene,disease"
    response = requests.get(url)
    time.sleep(0.5)
    doc = ElementTree.fromstring(response.content)
    tree = ElementTree.ElementTree(doc)
    root = tree.getroot()
    doc = root[3]
    passage = doc[1:]
    text_list, element_list, identifier_list, type_list = [], [], [], []
    for i in passage:
        for text in i.iterfind("text"):
            text = text.text
            text_list.append(text)
        for annotation in i.iterfind("annotation"):
            for text in annotation.iterfind("text"):
                element = text.text
                element_list.append(element)
            infos = annotation.findall("infon")
            try:
                identifier = infos[0].text
            except Exception:
                identifier = ""
            identifier_list.append(identifier.replace("MESH:", ""))
            try:
                typex = infos[1].text
            except Exception:
                typex = infos[0].text
            type_list.append(typex)
    text_list = [i.strip() for i in text_list]
    element_list = [i.strip() for i in element_list]
    identifier_list = [i.strip() for i in identifier_list]
    type_list = [i.strip() for i in type_list]
    df = pd.DataFrame(
        data=zip(element_list, identifier_list, type_list),
        columns=["element", "identifier", "type"],
    )
    df.identifier = df.identifier.replace("Disease", np.nan)
    df = df.drop_duplicates("identifier")
    disease_df = df[df.type == "Disease"]
    gene_df = df[df.type == "Gene"]
    disease_df = disease_df.drop("type", axis=1)
    gene_df = gene_df.drop("type", axis=1)
    a = list(gene_df[["element", "identifier"]].itertuples(index=False, name=None))
    b = list(disease_df[["element", "identifier"]].itertuples(index=False, name=None))
    pairs = list(set([(i, j) for i in a for j in b]))
    gene_df.to_csv(f"output/{document_id}/genes.csv", encoding="utf-8", index=False)
    disease_df.to_csv(
        f"output/{document_id}/diseases.csv", encoding="utf-8", index=False
    )
    return gene_df, disease_df, pairs
