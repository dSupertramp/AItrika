from engine.aitrika import OnlineAItrika, LocalAItrika
from llm.groq import GroqLLM
from utils.text_parser import generate_documents
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()
    pubmed_id = 23747889
    aitrika_engine = OnlineAItrika(pubmed_id=pubmed_id)
    abstract = aitrika_engine.abstract()
    associations = aitrika_engine.associations()
    print(associations)

    ## Prepare the documents
    documents = generate_documents(content=abstract)

    ## Set the LLM
    llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))

    ## Query your document
    query = "Is BRCA1 associated with breast cancer?"
    print(llm.query_model(query=query))
