from parser.article_parser import (
    parse_article,
    pubtator,
    get_pairs,
    extract_mesh_terms,
    extract_other_terms,
)

from parser.cohere import get_associations

# from parser.openai import get_associations


if __name__ == "__main__":
    document_id = "32819603"
    paper_id, title, abstract, document = parse_article(document_id=document_id)
    gene_df, disease_df = pubtator(document_id=document_id)
    mesh_terms = extract_mesh_terms(document_id=document_id)
    other_terms = extract_other_terms(document_id=document_id)
    pairs = get_pairs(gene_df, disease_df)
    ## result_cohere = get_associations(document=document, pairs=pairs)
    ## result_openai = get_associations(document=document, pairs=pairs)
