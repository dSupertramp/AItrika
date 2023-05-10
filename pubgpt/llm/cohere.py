from typing import List, Tuple
from dotenv import load_dotenv
import cohere
import os


load_dotenv()


def get_associations(document: str, pairs: List[Tuple[str, str]]):
    temperature, max_tokens = (0, 500)
    gene_id, disease_id, disease_umls = ([] for _ in range(3))
    pre_prompt: list = []
    for index, item in enumerate(pairs, 1):
        pre_prompt.append(
            f"{index}) {item[0][0].strip()} associated with {item[1][0].strip()}?"
        )
    pre_prompt = "\n".join(pre_prompt)
    prompt = f"""
    According to this abstract:\n
    {document}\n
    Can you tell me if:\n
    {pre_prompt}\n
    As result, provide me a CSV with:
    - Boolean result (only 'Yes' or 'No')
    - The parte before "is associated with"
    - The part after "is associated with"
    For instance:
    'Yes,X,Y'
    """.strip()
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = (
        co.generate(
            model="command-xlarge-nightly",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        .generations[0]
        .text
    )
    print(response)
