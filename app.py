"""
Streamlit app for demo.
"""

import streamlit as st
from aitrika.online.online_aitrika import OnlineAItrika
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


def format_list_to_markdown(items):
    markdown_text = ""
    if items[0].strip().startswith("**") and items[0].strip().endswith("**"):
        markdown_text += f"## {items[0].strip('* ')}\n\n"
        items = items[1:]
    for item in items:
        parts = item.split("\n\n")
        if len(parts) >= 2:
            for subitem in parts[1:]:
                markdown_text += f"- {subitem.strip()}\n"
        else:
            markdown_text += f"- {item.strip()}\n"
            markdown_text = markdown_text.replace('"', "")
        markdown_text += "\n"
    return markdown_text


pubmed_id = st.text_input("Enter the PubMed ID", placeholder="23747889")

if pubmed_id:
    engine = OnlineAItrika(pubmed_id=pubmed_id)
    documents = generate_documents(content=engine.extract_abstract())
    llm = GroqLLM(documents=documents, api_key=os.getenv("GROQ_API_KEY"))
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
                "Paper results",
                "Number of participants",
                "Characteristics of participants",
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
            methods = engine.extract_methods(llm=llm).split("---")
            formatted_methods = format_list_to_markdown(methods)
            st.markdown(formatted_methods)

        elif option == "Acknowledgements":
            st.markdown("## Acknowledgements")
            st.write(engine.extract_acknowledgements(llm=llm))
        elif option == "Introduction":
            st.markdown("## Introduction")
            st.write(engine.extract_introduction(llm=llm))
        elif option == "Number of participants":
            st.markdown("## Number of participants")
            st.write(engine.extract_number_of_participants(llm=llm))

        elif option == "Paper results":
            st.markdown("## Paper results")
            paper_results = engine.extract_paper_results(llm=llm).split("---")
            formatted_paper_results = format_list_to_markdown(paper_results)
            st.markdown(formatted_paper_results)

        elif option == "Characteristics of participants":
            st.markdown("## Characteristics of participants")
            characteristics_of_participants = (
                engine.extract_characteristics_of_participants(llm=llm).split("---")
            )
            formatted_characteristics_of_participants = format_list_to_markdown(
                characteristics_of_participants
            )
            st.markdown(formatted_characteristics_of_participants)
