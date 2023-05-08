import pandas as pd
import numpy as np
from Bio import Entrez
from Bio import Medline
from Bio import SeqIO
import requests
import re
import nltk
import json
import time
from itertools import cycle, islice
from xml.etree import ElementTree
import xmltodict


def parse_document(document_ID):
    query = str(document_ID)

    def search(TERM):
        Entrez.email = 'random@example.com'
        handle = Entrez.esearch(
            db='pubmed', sort='relevance', retmax=1, retmode='text', term=TERM)
        result = Entrez.read(handle)
        ids = result['IdList']
        handle = Entrez.efetch(db='pubmed', sort='relevance',
                               retmode='text', rettype='medline', id=ids)
        records = Medline.parse(handle)
        return records

    for record in search(query):
        title = record.get('TI', '')
        abstract = record.get('AB', '')
        ID = record.get('PMID', '')
        mesh = record.get('MH', '')
        other_terms = record.get('OT', '')

    document = title + ' ' + abstract
    with open('document.txt', 'w') as f:
        f.write(document)
    return ID, title, abstract, document


def pubtator(document_ID):
    FORMAT = 'biocxml'
    TYPE = "pmids"
    BIOCONCEPTS = 'gene,disease'
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/{FORMAT}?{TYPE}={document_ID}&concepts={BIOCONCEPTS}"
    response = requests.get(url)
    time.sleep(1.5)
    doc = ElementTree.fromstring(response.content)
    tree = ElementTree.ElementTree(doc)
    tree.write('content.xml', encoding='utf-8')
    root = tree.getroot()
    doc = root[3]
    passage = doc[1:]
    TEXT = []
    ELEMENT = []
    IDENTIFIER = []
    TYPE = []
    for i in passage:
        for text in i.iterfind('text'):
            text = text.text
            TEXT.append(text)
        for annotation in i.iterfind('annotation'):
            for text in annotation.iterfind('text'):
                element = text.text
                ELEMENT.append(element)
            infos = annotation.findall('infon')
            try:
                identifier = infos[0].text
            except:
                identifier = ''
            IDENTIFIER.append(identifier.replace('MESH:', ''))
            try:
                typex = infos[1].text
            except:
                typex = infos[0].text
            TYPE.append(typex)
    df = pd.DataFrame(data=zip(ELEMENT, IDENTIFIER, TYPE),
                      columns=['element', 'identifier', 'type'])
    df.identifier = df.identifier.replace('Disease', np.nan)
    df = df.drop_duplicates('identifier')
    disease_df = df[df.type == 'Disease']
    gene_df = df[df.type == 'Gene']
    gene_df.to_csv('genes_from_pubtator.csv', encoding='utf-8', index=False)
    disease_df.to_csv('diseases_from_pubtator.csv',
                      encoding='utf-8', index=False)
    return gene_df, disease_df
