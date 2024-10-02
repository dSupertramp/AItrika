from fastapi import FastAPI, Body
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


class QueryRequest(BaseModel):  # {{ edit_1 }}
    pubmed_id: int
    query: str


@app.post("/associations")
def get_associations(request: PubMedRequest):
    engine = OnlineAItrika(pubmed_id=request.pubmed_id)
    associations = engine.extract_associations()
    return {"associations": associations}


@app.post("/abstract")
def get_abstract(request: PubMedRequest):
    engine = OnlineAItrika(pubmed_id=request.pubmed_id)
    abstract = engine.extract_abstract()
    return {"abstract": abstract}


@app.post("/query")
def query_document(request: QueryRequest):
    engine = OnlineAItrika(pubmed_id=request.pubmed_id)
    abstract = engine.extract_abstract()
    documents = generate_documents(content=abstract)
    llm = GroqLLM(
        documents=documents,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.query(query=request.query)
    return {"result": result}


@app.post("/results")
def get_results(request: PubMedRequest):
    engine = OnlineAItrika(pubmed_id=request.pubmed_id)
    llm = GroqLLM(
        documents=generate_documents(content=engine.extract_abstract()),
        api_key=os.getenv("GROQ_API_KEY"),
    )
    results = engine.extract_results(llm=llm)
    return {"results": results}


@app.post("/participants")
def get_number_of_participants(request: PubMedRequest):
    engine = OnlineAItrika(pubmed_id=request.pubmed_id)
    llm = GroqLLM(
        documents=generate_documents(content=engine.extract_abstract()),
        api_key=os.getenv("GROQ_API_KEY"),
    )
    number_of_participants = engine.extract_number_of_participants(llm=llm)
    return {"number_of_participants": number_of_participants}


@app.post("/outcomes")
def get_outcomes(request: PubMedRequest):
    engine = OnlineAItrika(pubmed_id=request.pubmed_id)
    llm = GroqLLM(
        documents=generate_documents(content=engine.extract_abstract()),
        api_key=os.getenv("GROQ_API_KEY"),
    )
    outcomes = engine.extract_outcomes(llm=llm)
    return {"outcomes": outcomes}


if __name__ == "__main__":
    with open("docs/api/openapi.json", "w") as f:
        json.dump(
            get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                description=app.description,
                routes=app.routes,
                # openapi_prefix=app.openapi_prefix,
            ),
            f,
        )
    uvicorn.run(app, host="0.0.0.0", port=8000)
