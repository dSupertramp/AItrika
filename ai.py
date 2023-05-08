import pandas as pd
import numpy as np
from Bio import Entrez
from Bio import Medline
from Bio import SeqIO
import requests
import re
import time
import sys
from itertools import cycle, islice
from xml.etree import ElementTree
import xmltodict
import openai
from utilities import parse_document, pubtator
import logging
import logging.config


# LOGGING
logging.basicConfig(handlers=[
    logging.FileHandler("debug.log"),
    logging.StreamHandler()
],
    level=logging.DEBUG,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


def get_pairs(gene_df, disease_df):
    logging.info("Getting pairs")
    a = list(gene_df[['element', 'identifier']
                     ].itertuples(index=False, name=None))
    b = list(disease_df[['element', 'identifier']
                        ].itertuples(index=False, name=None))
    pairs = list(set([(i, j) for i in a for j in b]))
    return pairs


def get_associations(document, document_id, pairs):
    logging.info(f"Using OpenAI API for document with Pubmed ID {document_id}")
    auth_params = {"email": "salvatoredanilopalumbo@gmail.com",
                   "password": "PVFXYXpkV1p1VkdkeUVETTFrak4wVm1ibGQyY3BSR1JCNVVTTTkwVE1sa1RCUkVacE4zWmw1V1owWlRPMUFUTXlRWFp1VjJaemxHWg=="}
    api_host = "https://www.disgenet.org/api"
    VOCABULARY = 'mesh'
    SOURCE = 'CURATED'

    openai.api_key = "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"

    # engine = "text-davinci-001"
    engine = 'text-curie-001'

    temperature, frequency_penalty, presence_penalty = 0, 0, 0
    max_tokens = 500
    top_p = 1

    DATA = pd.DataFrame(columns=['geneId', 'diseaseId'])
    gene_id, disease_id, disease_umls = ([] for _ in range(3))

    for j in pairs:
        prompt = f"According to this text:\n\n{document}\n\nIs {j[0][0]} associated with {j[1][0]}?"
        response = openai.Completion.create(engine=engine, prompt=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty,
                                            presence_penalty=presence_penalty)
        logging.info("Waiting 1.5 sec for OpenAI API")
        time.sleep(1.5)
        for i in response['choices']:
            # print(i)
            p = str(i['text'])
            p = p.replace('\n', '')
            if p.startswith('Yes'):
                gene_id.append(j[0][1])
                disease_id.append(j[1][1])
                # print(f'{j[0][0]} - {j[0][1]}')
                # print(f'{j[1][0]} - {j[1][1]}')
    DATA.geneId = gene_id
    DATA.diseaseId = disease_id
    return DATA


if __name__ == "__main__":
    logging.info("Executing main on AI.py")
    document_ID = sys.argv[1]
    ID, title, abstract, document = parse_document(document_ID)
    gene_df, disease_df = pubtator(document_ID)
    pairs = get_pairs(gene_df, disease_df)
    result = check_associations(document, ID, pairs)
    logging.info(f"Document ID: {document_ID}")
    print(f"Document ID: {document_ID}")
    print('Results: ')
    print(result)
    result.to_csv('result.csv', encoding='utf-8', index=False)
