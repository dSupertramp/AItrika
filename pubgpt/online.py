from online_parser.article_parser import (
    parse_article,
    extract_genes_and_diseases,
    extract_mesh_terms,
    extract_other_terms,
)


# from llm.openai import get_associations
from llm_parser.cohere import get_associations

# from llm_parser.starcoder import get_associations


if __name__ == "__main__":
    pubmed_id = "32819603"
    paper_id, title, abstract, document = parse_article(pubmed_id=pubmed_id)
    gene_df, disease_df, pairs = extract_genes_and_diseases(pubmed_id=pubmed_id)
    mesh_terms = extract_mesh_terms(pubmed_id=pubmed_id)
    other_terms = extract_other_terms(pubmed_id=pubmed_id)
    ######################################################

    # result_openai, result_cohere

    result = get_associations(document=document, pubmed_id=pubmed_id, pairs=pairs)

    print(result)
