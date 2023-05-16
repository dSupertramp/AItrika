from typing import List, Tuple
from dotenv import load_dotenv
import openai
import os


load_dotenv()


def get_associations(document: str, document_id: str, pairs: List[Tuple[str, str]]):
    temperature, frequency_penalty, presence_penalty, max_tokens, top_p, engine = (
        0,
        0,
        0,
        500,
        1,
        "gpt-3.5-turbo",
    )
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
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )["choices"]
    with open(f"output/{document_id}/openai_results.csv", "w") as f:
        f.write("result,gene,disease")
        f.write(response)
    return response
