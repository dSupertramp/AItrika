import streamlit as st

from online_parser.article_parser import (
    parse_article,
    extract_genes_and_diseases,
    extract_mesh_terms,
    extract_other_terms,
)
from pdf_parser.utils import read_pdf, extract_pdf_content, split_pdf_content


## PDF PARSER
# from pdf_parser.openai import create_embeddings_openai, retriever_openai
# from pdf_parser.cohere import create_embeddings_cohere, retriever_cohere
# from pdf_parser.starcoder import create_embeddings, retriever
from pdf_parser.falcon import create_embeddings, retriever


## LLM
# from llm.openai import get_associations, summarize
# from llm_parser.cohere import get_associations, summarize
# from llm_parser.starcoder import get_associations, summarize
# from llm_parser.falcon import get_associations, summarize
# from llm_parser.llama2 import get_associations, summarize
# from llm_parser.ollama import get_associations, summarize
from llm_parser.falcon import get_associations, summarize

st.set_page_config(page_title="PubGPT", initial_sidebar_state="auto")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}          
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("PubGPT 💉📄")


@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")


def online_parser():
    st.markdown("""## Online parser""")
    pubmed_id = st.text_input("PubMed ID", "32819603")
    parse_paper = st.button("Parse paper")
    if parse_paper:
        paper_id, title, abstract, document = parse_article(pubmed_id=pubmed_id)
        st.write(f"Paper ID: {paper_id}")
        st.write(f"Title: {title}")
        st.write(f"Abstract: {abstract}")
        gene_df, disease_df, pairs = extract_genes_and_diseases(pubmed_id=pubmed_id)
        mesh_terms = extract_mesh_terms(pubmed_id=pubmed_id)
        other_terms = extract_other_terms(pubmed_id=pubmed_id)
        first_row = st.columns(2)
        first_row[0].markdown("### Genes")
        first_row[0].dataframe(gene_df)
        st.download_button(
            label="Download genes as CSV",
            data=convert_df(df=gene_df),
            file_name=f"{pubmed_id}_genes.csv",
        )
        first_row[1].markdown("### Diseases")
        first_row[1].dataframe(disease_df)
        st.download_button(
            label="Download diseases as CSV",
            data=convert_df(df=disease_df),
            file_name=f"{pubmed_id}_diseases.csv",
        )

        second_row = st.columns(2)
        if mesh_terms is not None:
            second_row[0].markdown("### MeSH terms")
            second_row[0].dataframe(mesh_terms)
            st.download_button(
                label="Download MeSH terms as CSV",
                data=convert_df(df=mesh_terms),
                file_name=f"{pubmed_id}_mesh_terms.csv",
            )

        if other_terms is not None:
            second_row[1].markdown("### Other terms")
            second_row[1].dataframe(other_terms)
            st.download_button(
                label="Download other terms as CSV",
                data=convert_df(df=other_terms),
                file_name=f"{pubmed_id}_other_terms.csv",
            )
    extract_associations = st.button("Extract associations between genes and diseases")
    st.warning("May produce incorrect informations", icon="⚠️")
    if extract_associations:
        paper_id, title, abstract, document = parse_article(pubmed_id=pubmed_id)
        gene_df, disease_df, pairs = extract_genes_and_diseases(pubmed_id=pubmed_id)
        result = get_associations(document=document, pubmed_id=pubmed_id, pairs=pairs)
        st.write(result)


def local_parser():
    st.markdown("""## Local parser""")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        pdf = read_pdf(uploaded_file)
        pdf_content = extract_pdf_content(pdf=pdf)
        splitted_text_from_pdf = split_pdf_content(
            pdf_content=pdf_content, chunk_size=1000, chunk_overlap=200
        )
        embeddings = create_embeddings(splitted_text_from_pdf=splitted_text_from_pdf)
        query = st.text_input(
            "Insert query here:",
            "Es: Is BRCA1 associated with breast cancer?",
        )
        if st.button("Query document"):
            st.write(retriever(query=query, embeddings=embeddings))


if __name__ == "__main__":
    online_parser()
    local_parser()
