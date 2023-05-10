from typing import List, Tuple
from dotenv import load_dotenv
from parser.article_parser import (
    parse_article,
    pubtator,
    get_pairs,
)
import openai
import pandas as pd
import os


load_dotenv()


def get_associations(document: str, pairs: List[Tuple[str, str]]):
    temperature, frequency_penalty, presence_penalty, max_tokens, top_p, engine = (
        0,
        0,
        0,
        500,
        1,
        "text-curie-001",
    )
    data = pd.DataFrame(columns=["geneId", "diseaseId"])
    gene_id, disease_id, disease_umls = ([] for _ in range(3))
    pre_prompt: list = []
    for index, item in enumerate(pairs, 1):
        pre_prompt.append(
            f"{index}) {item[0][0].strip()} associated with {item[1][0].strip()}?"
        )
    pre_prompt = f"\n".join(pre_prompt)
    prompt = f"""
    According to this abstract: \n
    {document} \n
    Can you tell me if: \n
    {pre_prompt} \n
    Give me each answer in a bullet list, and provide me a boolean result (only 'Yes' or 'No')
    """.strip()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    return response["choices"]
