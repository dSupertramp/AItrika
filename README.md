# Cura

![Cura](images/logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Enhance your knowledge in medical research.

Cura is a tool that can extract lots of relevant informations inside medical papers in an easy way:

- Abstract
- Full text (when available)
- Genes
- Diseases
- Associations between genes and diseases (powered by LLMs and Llamaindex)
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
from engine.cura import OnlineCura
cura_engine = OnlineCura(pubmed_id=pubmed_id)
title = cura_engine.get_title()
print(title)
```

Or you can parse a local pdf:

```python
from engine.cura import LocalCura
cura_engine = LocalCura(pdf_path = pdf_path)
title = cura_engine.get_title()
print(title)
```

```
Breast cancer genes: beyond BRCA1 and BRCA2.
```

You can get other informations, like the associations between genes and diseases:

```python
## Prepare the documents
documents = generate_documents(content=abstract)
llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))
associations = cura_engine.associations(llm=llm)
print(associations)
```

```
gene_id,gene_name,disease_id,disease_name,is_associated
672,BRCA1,MESH:D001943,Breast Neoplasms,True
672,BRCA1,MESH:D009386,Neoplastic Syndromes Hereditary,True
...
```

## License

Cura is licensed under the MIT License. See the LICENSE file for more details.
