"""
Main script for demo.
"""

from aitrika.engine.aitrika import OnlineAItrika
from aitrika.llm.groq import GroqLLM
from aitrika.utils.text_parser import generate_documents
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()
    pubmed_id = 23747889
    engine = OnlineAItrika(pubmed_id=pubmed_id)
    abstract = engine.extract_abstract()
    associations = engine.extract_associations()
    print(associations)

    ## Prepare the documents (you can use full-text if available)
    documents = generate_documents(content=abstract)

    ## Set the LLM (default for Groq is llama3-70b-8192)
    llm = GroqLLM(
        documents=documents,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    ## Query your document
    query = "Is BRCA1 associated with breast cancer?"
    print(llm.query(query=query))

    ## Extract paper results
    results = engine.extract_results(llm=llm)
    print(results)
