from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from aitrika.online.online_aitrika import OnlineAItrika
from aitrika.llm.groq import GroqLLM
from aitrika.utils.text_parser import generate_documents
from dotenv import load_dotenv
from fastapi.openapi.utils import get_openapi
import json
import os
import uvicorn

app = FastAPI(title="AItrika API", description="AItrika API", version="0.1.0")
load_dotenv()


class PubMedRequest(BaseModel):
    pubmed_id: int


class QueryRequest(BaseModel):
    pubmed_id: int
    query: str


@app.post("/associations", summary="Get associations from a PubMed article")
def get_associations(request: PubMedRequest):
    """
    Extracts associations from a PubMed article based on its ID.

    Args:
        request (PubMedRequest): An object containing the PubMed ID.

    Returns:
        dict: A dictionary containing the extracted associations.

    Raises:
        HTTPException: If the PubMed ID is invalid or the article is not found.
    """
    try:
        engine = OnlineAItrika(pubmed_id=request.pubmed_id)
        associations = engine.extract_associations()
        return {"associations": associations}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/abstract", summary="Get abstract of a PubMed article")
def get_abstract(request: PubMedRequest):
    """
    Retrieves the abstract of a PubMed article based on its ID.

    Args:
        request (PubMedRequest): An object containing the PubMed ID.

    Returns:
        dict: A dictionary containing the abstract of the article.

    Raises:
        HTTPException: If the PubMed ID is invalid or the article is not found.
    """
    try:
        engine = OnlineAItrika(pubmed_id=request.pubmed_id)
        abstract = engine.extract_abstract()
        return {"abstract": abstract}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/query", summary="Query a PubMed article")
def query_document(request: QueryRequest):
    """
    Queries a PubMed article with a user-provided question.

    Args:
        request (QueryRequest): An object containing the PubMed ID and the query.

    Returns:
        dict: A dictionary containing the result of the query.

    Raises:
        HTTPException: If the PubMed ID is invalid, the article is not found, or the query fails.
    """
    try:
        engine = OnlineAItrika(pubmed_id=request.pubmed_id)
        abstract = engine.extract_abstract()
        documents = generate_documents(content=abstract)
        llm = GroqLLM(
            documents=documents,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        result = llm.query(query=request.query)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/results", summary="Get results from a PubMed article")
def get_results(request: PubMedRequest):
    """
    Extracts results from a PubMed article based on its ID.

    Args:
        request (PubMedRequest): An object containing the PubMed ID.

    Returns:
        dict: A dictionary containing the extracted results.

    Raises:
        HTTPException: If the PubMed ID is invalid or the article is not found.
    """
    try:
        engine = OnlineAItrika(pubmed_id=request.pubmed_id)
        llm = GroqLLM(
            documents=generate_documents(content=engine.extract_abstract()),
            api_key=os.getenv("GROQ_API_KEY"),
        )
        results = engine.extract_results(llm=llm)
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/participants", summary="Get number of participants from a PubMed article")
def get_number_of_participants(request: PubMedRequest):
    """
    Extracts the number of participants from a PubMed article based on its ID.

    Args:
        request (PubMedRequest): An object containing the PubMed ID.

    Returns:
        dict: A dictionary containing the number of participants.

    Raises:
        HTTPException: If the PubMed ID is invalid or the article is not found.
    """
    try:
        engine = OnlineAItrika(pubmed_id=request.pubmed_id)
        llm = GroqLLM(
            documents=generate_documents(content=engine.extract_abstract()),
            api_key=os.getenv("GROQ_API_KEY"),
        )
        number_of_participants = engine.extract_number_of_participants(llm=llm)
        return {"number_of_participants": number_of_participants}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/outcomes", summary="Get outcomes from a PubMed article")
def get_outcomes(request: PubMedRequest):
    """
    Extracts outcomes from a PubMed article based on its ID.

    Args:
        request (PubMedRequest): An object containing the PubMed ID.

    Returns:
        dict: A dictionary containing the extracted outcomes.

    Raises:
        HTTPException: If the PubMed ID is invalid or the article is not found.
    """
    try:
        engine = OnlineAItrika(pubmed_id=request.pubmed_id)
        llm = GroqLLM(
            documents=generate_documents(content=engine.extract_abstract()),
            api_key=os.getenv("GROQ_API_KEY"),
        )
        outcomes = engine.extract_outcomes(llm=llm)
        return {"outcomes": outcomes}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


if __name__ == "__main__":
    with open("docs/api-reference/openapi.json", "w") as f:
        json.dump(
            get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                description=app.description,
                routes=app.routes,
            ),
            f,
        )
    uvicorn.run(app, host="0.0.0.0", port=8000)
