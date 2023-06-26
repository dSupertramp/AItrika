from typing import Tuple
from Bio import Entrez, Medline
import pandas as pd
import numpy as np
import requests
import time
from xml.etree import ElementTree
from utils.utils import create_log_folder


def search_on_pubmed(pubmed_id: str) -> list:
    """
    Search on PubMed using a PubMed ID.

    Args:
        pubmed_id (str): PubMed ID

    Returns:
        list: List of records of the article
    """
    Entrez.email = "random@example.com"
    handle = Entrez.esearch(
        db="pubmed", sort="relevance", retmax=1, retmode="text", term=pubmed_id
    )
    result = Entrez.read(handle)
    ids = result["IdList"]
    handle = Entrez.efetch(
        db="pubmed", sort="relevance", retmode="text", rettype="medline", id=ids
    )
    records = Medline.parse(handle)
    return records


def parse_article(pubmed_id: str) -> Tuple[str, str, str, str]:
    """
    Parse the article.

    Args:
        pubmed_id (str): PubMed ID

    Returns:
        Tuple[str, str, str, str]: PubMed ID, title, abstract, title+abstract
    """
    create_id_folder(pubmed_id=pubmed_id)
    for record in search_on_pubmed(pubmed_id=pubmed_id):
        title = record.get("TI", "")
        abstract = record.get("AB", "")
        pubmed_id = record.get("PMID", "")
    document = title + " " + abstract
    with open(f"output/{pubmed_id}/document.txt", "w") as f:
        f.write(document)
    return pubmed_id, title, abstract, document


def extract_mesh_terms(pubmed_id: str) -> pd.DataFrame:
    """
    Extract MeSH terms from article.

    Args:
        pubmed_id (str): PubMed ID

    Returns:
        pd.DataFrame: DataFrame with MeSH terms
    """
    create_id_folder(pubmed_id=pubmed_id)
    for record in search_on_pubmed(pubmed_id=pubmed_id):
        mesh_terms = record.get("MH", "")
    df = pd.DataFrame(data=zip(mesh_terms), columns=["element"])
    df.to_csv(f"output/{pubmed_id}/mesh_terms.csv", encoding="utf-8", index=False)


def extract_other_terms(pubmed_id: str) -> pd.DataFrame:
    """
    Extract other terms from article.

    Args:
        pubmed_id (str): PubMed ID

    Returns:
        pd.DataFrame: DataFrame with other terms
    """
    create_id_folder(pubmed_id=pubmed_id)
    for record in search_on_pubmed(pubmed_id=pubmed_id):
        other_terms = record.get("OT", "")
    df = pd.DataFrame(data=zip(other_terms), columns=["element"])
    df.to_csv(f"output/{pubmed_id}/other_terms.csv", encoding="utf-8", index=False)


def extract_genes_and_diseases(pubmed_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract genes and disease from article.

    Args:
        pubmed_id (str): PubMed ID

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with genes and DataFrame with diseases
    """
    create_id_folder(pubmed_id=pubmed_id)
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids={pubmed_id}&concepts=gene,disease"
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
    genes_pairs = list(
        gene_df[["element", "identifier"]].itertuples(index=False, name=None)
    )
    disease_pairs = list(
        disease_df[["element", "identifier"]].itertuples(index=False, name=None)
    )
    pairs = list(set([(i, j) for i in genes_pairs for j in disease_pairs]))
    gene_df.to_csv(f"output/{pubmed_id}/genes.csv", encoding="utf-8", index=False)
    disease_df.to_csv(f"output/{pubmed_id}/diseases.csv", encoding="utf-8", index=False)
    return gene_df, disease_df, pairs


def extract_chemicals(pubmed_id: str):
    create_id_folder(pubmed_id=pubmed_id)
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids={pubmed_id}&concepts=chemical"
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
    df["element"] = df["element"].astype(str)
    df["identifier"] = df["identifier"].astype(str)
    df["type"] = df["type"].astype(str)
    df = df.drop_duplicates("identifier")
    df.to_csv(f"output/{pubmed_id}/chemicals.csv", sep=",", index=False)
    return df


def extract_mutations(pubmed_id: str):
    create_id_folder(pubmed_id=pubmed_id)
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids={pubmed_id}&concepts=mutation"
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
    df["element"] = df["element"].astype(str)
    df["identifier"] = df["identifier"].astype(str)
    df["type"] = df["type"].astype(str)
    df = df.drop_duplicates("identifier")
    df.to_csv(f"output/{pubmed_id}/mutation.csv", sep=",", index=False)
    return df


def extract_species(pubmed_id: str):
    create_id_folder(pubmed_id=pubmed_id)
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids={pubmed_id}&concepts=species"
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
    df["element"] = df["element"].astype(str)
    df["identifier"] = df["identifier"].astype(str)
    df["type"] = df["type"].astype(str)
    df = df.drop_duplicates("identifier")
    df.to_csv(f"output/{pubmed_id}/species.csv", sep=",", index=False)
    return df
