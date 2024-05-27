"""
Streamlit app for demo.
"""

import streamlit as st
from aitrika.engine.aitrika import LocalAItrika
from aitrika.utils.text_parser import generate_documents
from aitrika.llm.groq import GroqLLM
from dotenv import load_dotenv
import os
import time

load_dotenv()

st.set_page_config(
    page_title="AItrika",
    page_icon="images/logo.png",
)
st.title("AItrika 🧪")


def response_generator(query: str):
    response = llm.query(query=query)
    print(response)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None


uploaded_file = st.file_uploader(
    "Choose your .pdf file (put the file in the same folder)", type="pdf"
)
if uploaded_file is not None:
    pdf_path = uploaded_file.name
    engine = LocalAItrika(pdf_path=pdf_path)
    documents = generate_documents(content=engine.extract_full_text())
    llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))


## CHAT
if uploaded_file is not None:
    with st.expander("Query"):
        query = st.text_input("Insert your query: ")
        st.write(response_generator(query=query))


## EXTRACTOR
if uploaded_file is not None:
    with st.expander("Select the information that you want to extract: "):
        option = st.selectbox(
            " ",
            (
                "PubMed ID",
                "Title",
                "Abstract",
                "Authors",
                "Full text",
                "Journal",
                "Genes",
                "Diseases",
                "Species",
                "Chemicals",
                "Mutations",
                "Associations between genes and diseases",
                "Results",
                "Bibliography",
                "Methods",
                "Acknowledgements",
                "Introduction",
            ),
        )
        if option == "PubMed ID":
            st.markdown("## PubMed ID")
            st.write(engine.extract_pubmed_id())
        elif option == "Title":
            st.markdown("## Title")
            st.write(engine.extract_title())
        elif option == "Abstract":
            st.markdown("## Abstract")
            st.write(engine.extract_abstract())
        elif option == "Authors":
            st.markdown("## Authors")
            st.write(engine.extract_authors())
        elif option == "Full text":
            st.markdown("## Full text")
            st.write(engine.extract_full_text())
        elif option == "Journal":
            st.markdown("## Journal")
            st.write(engine.extract_journal())
        elif option == "Genes":
            st.markdown("## Genes")
            st.dataframe(engine.extract_genes(dataframe=True))
            st.json(engine.genes())
        elif option == "Diseases":
            st.markdown("## Diseases")
            st.dataframe(engine.extract_diseases(dataframe=True))
            st.json(engine.diseases())
        elif option == "Associations between genes and diseases":
            st.markdown("## Associations between genes and diseases")
            st.dataframe(engine.extract_associations(dataframe=True))
            st.json(engine.extract_associations())
        elif option == "Species":
            st.markdown("## Species")
            st.dataframe(engine.extract_species(dataframe=True))
            st.json(engine.extract_species())
        elif option == "Mutations":
            st.markdown("## Mutations")
            st.dataframe(engine.extract_mutations(dataframe=True))
            st.json(engine.extract_mutations())
        elif option == "Chemicals":
            st.markdown("## Chemicals")
            st.dataframe(engine.extract_chemicals(dataframe=True))
            st.json(engine.extract_chemicals())
        elif option == "Results":
            st.markdown("## Results")
            st.write(engine.extract_results(llm=llm))
        elif option == "Bibliography":
            st.markdown("## Bibliography")
            st.write(engine.extract_bibliography(llm=llm))
        elif option == "Methods":
            st.markdown("## Methods")
            st.write(engine.extract_methods(llm=llm))
        elif option == "Acknowledgements":
            st.markdown("## Acknowledgements")
            st.write(engine.extract_acknowledgements(llm=llm))
        elif option == "Introduction":
            st.markdown("## Introduction")
            st.write(engine.extract_introduction(llm=llm))
