import requests
import json
from typing import List, Tuple


def get_associations(
    document: str, pubmed_id: str, pairs: List[Tuple[str, str]]
) -> str:
    gene_id, disease_id, disease_umls = ([] for _ in range(3))
    pre_prompt: list = []
    for index, item in enumerate(pairs, 1):
        pre_prompt.append(
            f"{index}) {item[0][0].strip()} associated with {item[1][0].strip()}?"
        )
    pre_prompt = "\n".join(pre_prompt)
    prompt = f"""
        According to this abstract:\n
    {document.strip()}\n
    Can you tell me if:\n
    {pre_prompt.strip()}\n
    As result, provide me only CSV with:
    - Boolean result (only 'Yes' or 'No')
    - The entire part before the sentence "is associated with"
    - The entire part after the sentence "is associated with"
    For instance:
    'Yes,X,Y'
    Also, remove the numbers list (like 1)) from the CSV
        """.strip()
    url = "http://localhost:11434/api/generate"
    model = "orca-mini"
    payload = {
        "model": f"{model}",
        "prompt": f"[INST] {prompt} [/INST]",
        "raw": True,
        "stream": False,
    }
    response = requests.post(url, json=payload)
    data = json.loads(response.text)
    response = data["response"]
    return response


def summarize(document: str, pubmed_id: str) -> str:
    prompt = f"""
Summarize this text, trying to keep all relevant informations:
{document.strip()}
    """
    url = "http://localhost:11434/api/generate"
    model = "orca-mini"
    payload = {
        "model": f"{model}",
        "prompt": f"[INST] {prompt} [/INST]",
        "raw": True,
        "stream": False,
    }
    response = requests.post(url, json=payload)
    data = json.loads(response.text)
    response = data["response"]
    return response
