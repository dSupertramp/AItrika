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
    print(abstract)

    ## Prepare the documents
    documents = generate_documents(content=abstract)
    llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))
    associations = aitrika_engine.associations(llm=llm)
    print(associations)
