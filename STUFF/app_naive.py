from flask import Flask, jsonify, request, send_file, send_from_directory, render_template
import NAIVE
import utilities
import json
import os
app = Flask(__name__)


@app.route('/download_document')
def download_document():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    ID, title, abstract, document = NAIVE.parse_document(document_ID)
    with open('document.txt', 'w') as f:
        f.write(f"Pubmed ID: {document_ID}\n\n")
        f.write(document)
    return send_file('document.txt', as_attachment=True)


@app.route('/download_associations')
def download_associations():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    ID, title, abstract, document = NAIVE.parse_document(document_ID)
    gene_df, disease_df = NAIVE.pubtator(document_ID)
    result = NAIVE.process_text(document, gene_df, disease_df)
    result.to_csv('result.csv', encoding='utf-8', index=False)
    return send_file('result.csv', as_attachment=True)


@app.route('/download_genes')
def download_genes():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    gene_df, disease_df = NAIVE.pubtator(document_ID)
    gene_df.to_csv('genes_from_pubtator.csv', encoding='utf-8', index=False)
    return send_file('genes_from_pubtator.csv', as_attachment=True)


@app.route('/download_diseases')
def download_diseases():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    gene_df, disease_df = NAIVE.pubtator(document_ID)
    disease_df.to_csv('diseases_from_pubtator.csv',
                      encoding='utf-8', index=False)
    return send_file('diseases_from_pubtator.csv', as_attachment=True)


@app.route('/get_document')
def get_document():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    ID, title, abstract, document = NAIVE.parse_document(document_ID)
    print("OUTPUT: Document")
    return f"Pubmed ID: {document_ID}" + "<br>" + document


@app.route('/get_associations')
def get_associations():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    ID, title, abstract, document = NAIVE.parse_document(document_ID)
    gene_df, disease_df = NAIVE.pubtator(document_ID)
    result = NAIVE.process_text(document, gene_df, disease_df)
    result = result.to_json(orient='split')
    parsed = json.loads(result)
    print("OUTPUT: Associations")
    print(parsed)
    return parsed


@app.route('/get_genes')
def get_genes():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    gene_df, disease_df = NAIVE.pubtator(document_ID)
    result = gene_df.to_json()
    parsed = json.loads(result)
    print("OUTPUT: Pubtator Genes")
    print(parsed)
    return parsed


@app.route('/get_diseases')
def get_diseases():
    print('Running')
    document_ID = request.args.get('pubmed_id')
    gene_df, disease_df = NAIVE.pubtator(document_ID)
    result = disease_df.to_json()
    parsed = json.loads(result)
    print("OUTPUT: Pubtator Diseases")
    print(parsed)
    return parsed


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
