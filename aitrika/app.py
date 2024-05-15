"""
Streamlit app for demo.
"""

import streamlit as st
from aitrika.engine.aitrika import LocalAItrika
from utils.text_parser import generate_documents
from llm.groq import GroqLLM
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="AItrika",
    page_icon="images/logo.png",
)
st.title("AItrika")


uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")
if uploaded_file is not None:
    pdf_path = "data/" + uploaded_file.name
    engine = LocalAItrika(pdf_path=pdf_path)
    documents = generate_documents(content=engine.full_text())
    llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))
else:
    uploaded_file = None


query_button = st.button("Query")
if query_button and uploaded_file:
    query_document = st.text_input("Insert the query here 👇")
    if query_document:
        st.write(llm.query(query=query_document))

if uploaded_file:
    option = st.selectbox(
        "Select the information that you want to extract",
        (
            "PubMed ID",
            "Title",
            "Abstract",
            "Full text",
            "Genes",
            "Diseases",
            "Species",
            "Chemicals",
            "Mutations",
            "Associations between genes and diseases",
            "Results",
            "Bibliography",
            "Methods",
        ),
    )
    if option == "PubMed ID":
        st.markdown("## PubMed ID")
        st.write(engine.get_pubmed_id())
    elif option == "Title":
        st.markdown("## Title")
        st.write(engine.get_title())
    elif option == "Abstract":
        st.markdown("## Abstract")
        st.write(engine.abstract())
    elif option == "Full text":
        st.markdown("## Full text")
        st.write(engine.full_text())
    elif option == "Genes":
        st.markdown("## Genes")
        st.dataframe(engine.genes(dataframe=True))
        st.json(engine.genes())
    elif option == "Diseases":
        st.markdown("## Diseases")
        st.dataframe(engine.diseases(dataframe=True))
        st.json(engine.diseases())
    elif option == "Associations between genes and diseases":
        st.markdown("## Associations between genes and diseases")
        st.dataframe(engine.associations(dataframe=True))
        st.json(engine.associations())
    elif option == "Species":
        st.markdown("## Species")
        st.dataframe(engine.species(dataframe=True))
        st.json(engine.species())
    elif option == "Mutations":
        st.markdown("## Mutations")
        st.dataframe(engine.mutations(dataframe=True))
        st.json(engine.mutations())
    elif option == "Chemicals":
        st.markdown("## Chemicals")
        st.dataframe(engine.chemicals(dataframe=True))
        st.json(engine.chemicals())
    elif option == "Results":
        st.markdown("## Results")
        st.write(engine.results(llm=llm))
    elif option == "Bibliography":
        st.markdown("## Bibliography")
        st.write(engine.bibliography(llm=llm))
    elif option == "Methods":
        st.markdown("## Methods")
        st.write(engine.methods(llm=llm))
