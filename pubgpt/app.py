import streamlit as st

from parser.article_parser import (
    parse_article,
    extract_mesh_terms,
    extract_other_terms,
    extract_genes_and_diseases,
    extract_chemicals,
    extract_mutations,
    extract_species,
)


from llm.zephyr import create_embeddings, retriever, get_associations

# from llm.openai import create_embeddings, get_associations
# from llm.falcon import create_embeddings, retriever, get_associations


from llm.utils import read_document, read_pdf


st.set_page_config(page_title="PubGPT", initial_sidebar_state="auto", page_icon="💉")

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


def sidebar():
    st.sidebar.title("LLM selector")
    llm_choice = st.sidebar.selectbox(
        "Choose an LLM",
        ("OpenAI", "Falcon-7b", "Zephyr-7b-Alpha", "Cohere", "Starcoder"),
        index=None,
    )
    api_keys_mapping = {
        "OpenAI": ("OpenAI API Key", "OpenAI API Key"),
        "Cohere": ("Cohere API Key", "Cohere API Key"),
        "Falcon-7b": ("HuggingFace Hub API Key", "HuggingFace Hub API Key"),
        "Starcoder": ("HuggingFace Hub API Key", "HuggingFace Hub API Key"),
        "Zephyr-7b-Alpha": ("HuggingFace Hub API Key", "HuggingFace Hub API Key"),
    }

    if llm_choice in api_keys_mapping:
        label, placeholder = api_keys_mapping[llm_choice]
        api_key = st.sidebar.text_input(label=label, placeholder=placeholder)
        select_llm = st.sidebar.button("Select LLM")
        if llm_choice and api_key and select_llm:
            st.sidebar.info("LLM set!", icon="ℹ️")


def parse_paper(pubmed_id):
    paper_id, title, abstract, document = parse_article(pubmed_id=pubmed_id)
    st.markdown(f"**Paper ID**: {paper_id}")
    st.markdown(f"**Title**: {title}")
    st.markdown(f"**Abstract**: {abstract}")

    gene_df, disease_df, pairs = extract_genes_and_diseases(pubmed_id=pubmed_id)
    mesh_terms = extract_mesh_terms(pubmed_id=pubmed_id)
    other_terms = extract_other_terms(pubmed_id=pubmed_id)
    chemicals_terms = extract_chemicals(pubmed_id=pubmed_id)
    mutations_terms = extract_mutations(pubmed_id=pubmed_id)
    species_terms = extract_species(pubmed_id=pubmed_id)

    ##* FIRST ROW
    first_row = st.columns(2)
    first_row[0].markdown("### Genes")
    first_row[0].dataframe(
        gene_df,
        hide_index=True,
    )
    first_row[1].markdown("### Diseases")
    first_row[1].dataframe(
        disease_df,
        hide_index=True,
    )

    ##* SECOND ROW
    second_row = st.columns(2)
    second_row[0].markdown("### MeSH")
    second_row[0].dataframe(
        mesh_terms,
        hide_index=True,
    )
    second_row[1].markdown("### Chemicals")
    second_row[1].dataframe(
        chemicals_terms,
        hide_index=True,
    )

    ##* THIRD ROW
    third_row = st.columns(2)
    third_row[0].markdown("### Mutations")
    third_row[0].dataframe(
        mutations_terms,
        hide_index=True,
    )
    third_row[1].markdown("### Species")
    third_row[1].dataframe(
        species_terms,
        hide_index=True,
    )
    st.markdown("### Other terms")
    st.dataframe(
        other_terms,
        hide_index=True,
    )


def query_document(query, pubmed_id):
    paper_id, title, abstract, document = parse_article(pubmed_id=pubmed_id)
    document_content = read_document(
        document=document, chunk_size=1000, chunk_overlap=200
    )
    embeddings_query = create_embeddings(splitted_text=document_content)
    result = retriever(query=query, embeddings=embeddings_query)
    st.write(result)


def extract_associations(pubmed_id):
    st.warning("May produce incorrect informations", icon="⚠️")
    paper_id, title, abstract, document = parse_article(pubmed_id=pubmed_id)
    gene_df, disease_df, pairs = extract_genes_and_diseases(pubmed_id=pubmed_id)
    embeddings_associations = create_embeddings(splitted_text=document)
    result = get_associations(pairs=pairs, embeddings=embeddings_associations)
    st.write(result)


def online_parser():
    pubmed_id = st.text_input("PubMed ID", "32819603")
    query = st.text_input(
        label="Insert a query here:",
        placeholder="Es: Is BRCA2 associated with breast cancer?",
    )
    row = st.columns(3)
    parse_paper_button = row[0].button("Parse paper")
    query_doc_button = row[1].button("Query document")
    extract_associations_button = row[2].button("Extract associations")
    if parse_paper_button:
        parse_paper(pubmed_id)
    if query_doc_button:
        query_document(query=query, pubmed_id=pubmed_id)
    if extract_associations_button:
        extract_associations(pubmed_id)


def local_parser():
    st.markdown("""## Local parser""")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        pdf_content = read_pdf(
            pdf_path=uploaded_file, chunk_size=1000, chunk_overlap=200
        )
        embeddings_local_parser = create_embeddings(splitted_text=pdf_content)
        query_for_pdf = st.text_input(
            label="Insert query here:",
            placeholder="Es: Is BRCA1 associated with breast cancer?",
        )
        submit_query_for_pdf = st.button("Query PDF")
        if submit_query_for_pdf:
            st.write(retriever(query=query_for_pdf, embeddings=embeddings_local_parser))


if __name__ == "__main__":
    sidebar()
    online_parser()
    # local_parser()
