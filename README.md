# AItrika (formerly PubGPT)

![AItrika](images/logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Enhance your knowledge in medical research.

AItrika is a tool that can extract lots of relevant informations inside medical papers in an easy way:

- Abstract
- Full text (when available)
- Genes
- Diseases
- Associations between genes and diseases
- MeSH terms
- Other terms

And so on!

## 📦 Install

To install everything, you need `poetry`.
First of all, create a virtual environment with the command `python3 -m venv venv_name` and activate it with `source venv_name\bin\activate`.

After that, you can install poetry with the command `pip install poetry` and then run `poetry install`.

## 🔑 Set API Keys

In order to set API keys, type your keys into the `env.example` file and rename it to `.env`

## 🔍 Usage

You can easily get informations of a paper by passing a PubMed ID:

```python
from engine.aitrika import OnlineAItrika
aitrika_engine = OnlineAItrika(pubmed_id=pubmed_id)
title = aitrika_engine.get_title()
print(title)
```

Or you can parse a local pdf:

```python
from engine.aitrika import LocalAItrika
aitrika_engine = LocalAItrika(pdf_path = pdf_path)
title = aitrika_engine.get_title()
print(title)
```

```
Breast cancer genes: beyond BRCA1 and BRCA2.
```

You can get other informations, like the associations between genes and diseases:

```python
associations = aitrika_engine.associations()
```

```
[
  {
    "gene": "BRIP1",
    "disease": "Breast Neoplasms"
  },
  {
    "gene": "PTEN",
    "disease": "Breast Neoplasms"
  },
  {
    "gene": "CHEK2",
    "disease": "Breast Neoplasms"
  },
]
...
```

Or you can get a nice-formatted DataFrame:

```python
associations = aitrika_engine.associations(dataframe = True)
```

```
      gene                          disease
0    BRIP1                 Breast Neoplasms
1     PTEN                 Breast Neoplasms
2    CHEK2                 Breast Neoplasms
...
```

With the power of RAG, you can query your document:

```python
## Prepare the documents
documents = generate_documents(content=abstract)

## Set the LLM
llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))

## Query your document
query = "Is BRCA1 associated with breast cancer?"
print(llm.query_model(query=query))
```

```
The provided text suggests that BRCA1 is associated with breast cancer, as it is listed among the high-penetrance genes identified in family linkage studies as responsible for inherited syndromes of breast cancer.
```

## License

AItrika is licensed under the MIT License. See the LICENSE file for more details.

## TODO

- [ ] Create documentation
- [ ] Add docstrings
