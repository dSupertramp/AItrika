import pandas as pd
import numpy as np
from Bio import Entrez
from Bio import Medline
from Bio import SeqIO
import requests
import re
import nltk
import time
import sys
from itertools import cycle, islice
from xml.etree import ElementTree
import xmltodict
from utilities_naive import parse_document, pubtator
nltk.download('punkt')


def process_text(document, gene_df, disease_df):
    document = re.sub(r'[()]', '', document)
    document = re.sub(r'[\[\]]', '', document)
    document = document.replace('-->', '')
    document = document.replace('<--', '')
    document = document.replace('->', '')
    document = document.replace('<-', '')

    sentence = nltk.sent_tokenize(document)
    genes_discovered = gene_df.element.tolist()
    diseases_discovered = disease_df.element.tolist()

    genes_id = gene_df.identifier.tolist()
    diseases_id = disease_df.identifier.tolist()

    phrases = []
    for i in sentence:
        for j in genes_discovered:
            for k in diseases_discovered:
                if str(j) in i and str(k) in i and 'association' in i:
                    phrases.append(i)
    phrases = list(set(phrases))

    DATA = pd.DataFrame(columns=['geneId', 'diseaseId'])
    for i in phrases:
        genes = [j for j in genes_discovered if j in i]
        diseases = [j for j in diseases_discovered if j in i]
        gene_id = [gene_df['identifier']
                   [gene_df['element'] == i].values[0] for i in genes]
        disease_id = [disease_df['identifier']
                      [disease_df['element'] == i].values[0] for i in diseases]
        max_value = max(len(gene_id), len(disease_id))
        DATA = DATA.append({
            'geneId':  list(islice(cycle(gene_id), max_value)),
            'diseaseId': list(islice(cycle(disease_id), max_value)),
        }, ignore_index=True)
    DATA = DATA.explode(['geneId', 'diseaseId'])
    DATA = DATA.drop_duplicates()
    return DATA


if __name__ == "__main__":
    document_ID = sys.argv[1]
    ID, title, abstract, document = parse_document(document_ID)
    gene_df, disease_df = pubtator(document_ID)
    result = process_text(document, gene_df, disease_df)
    print(f"Document ID: {document_ID}")
    print('Results: ')
    print(result)
    result.to_csv('result.csv', encoding='utf-8', index=False)
