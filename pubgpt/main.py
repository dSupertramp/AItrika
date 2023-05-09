from parser.article_parser import parse_article, pubtator

if __name__ == "__main__":
    ID = "32819603"
    parse_article(document_id=ID)
    pubtator(document_id=ID)
