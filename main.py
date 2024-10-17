"""
Main script for demo.
"""

from aitrika.online.online_aitrika import OnlineAItrika
from aitrika.llm.openai import OpenAILLM
from aitrika.utils.text_parser import generate_documents
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    pubmed_id = 23747889
    engine = OnlineAItrika(pubmed_id=pubmed_id)

    # Extract the content
    abstract = engine.extract_abstract()

    # Prepare the documents (you can use full-text if available)
    documents = generate_documents(content=abstract)

    # Set the LLM (default for OpenAI is gpt-4o-mini)
    llm = OpenAILLM(
        documents=documents,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Query your document
    query = "Is BRCA1 associated with breast cancer?"
    print(llm.query(query=query))

    # Extract paper results
    results = engine.extract_results(llm=llm)
    print(results)

    # Extract number of participants
    number_of_participants = engine.extract_number_of_participants(llm=llm)
    print(number_of_participants)

    # Extract outcomes
    outcomes = engine.extract_outcomes(llm=llm)
    print(outcomes)
