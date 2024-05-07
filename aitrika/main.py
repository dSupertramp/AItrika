from engine.cura import OnlineCura, LocalCura
from llm.groq import GroqLLM
from utils.text_parser import generate_documents
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()
    pubmed_id = 23747889
    cura_engine = OnlineCura(pubmed_id=pubmed_id)
    abstract = cura_engine.abstract()
    print(abstract)

    ## Prepare the documents
    documents = generate_documents(content=abstract)
    llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))
    associations = cura_engine.associations(llm=llm)
    print(associations)
