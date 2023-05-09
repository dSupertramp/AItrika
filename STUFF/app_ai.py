from flask import Flask, request, send_file, render_template, jsonify, make_response
from flask_restx import Api, Resource
import jwt
import AI
import utilities
import json
import hashlib
from functools import wraps
import logging
import logging.config

# APP DEFINITION
app = Flask(__name__)


# TOKEN
# Token è il sha256 della stringa "token" (senza doppie apici, tutto in minuscolo)
TOKEN = '3c469e9d6c5875d37a43f353d4f88e61fcf812c66eee3457465a40b0da4153e0'

# AUTHORIZATIONS
authorizations = {
    'Bearer Auth': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    },
}

# TOKEN REQUIRED


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'X-API-KEY' in request.headers:
            token = request.headers['X-API-KEY']
        if not token:
            return {'message': 'Token is missing.'}, 403
        if token != TOKEN:
            return {'message': 'Token is wrong.'}, 403
        print('TOKEN: {}'.format(token))
        return f(*args, **kwargs)
    return decorated


# API CONFIGURATION
api = Api(app=app,
          version='1.0',
          title='Text mining API',
          authorizations=authorizations,
          security='Bearer Auth',
          description='Operations related to text-mining in medical literature, like gene-disease association extraction, document extraction, genes extraction and diseases extraction',)

# LOGGING
logging.basicConfig(handlers=[
    logging.FileHandler("debug.log"),
    logging.StreamHandler()
],
    level=logging.DEBUG,
    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# NAMESPACE DEFINITION
ns = api.namespace('text-mining')


# EXTRACT DOCUMENT ROUTE
@ns.route('/extract-document/<string:pubmed_id>')
@api.response(404, 'Pubmed ID not found or not valid.')
@api.response(200, 'Abstract content of the document with respective Pubmed ID')
@api.doc(params={'pubmed_id': {"description": "Pubmed ID of an article", "in": "query", "type": "string", "required": "true", "example": " 32819603"}})
class Document(Resource):
    @token_required
    def get(self, pubmed_id):
        app.logger.info("Extract document started")
        """Return abstract associated with Pubmed ID"""
        print('Running document extraction')
        document_ID = pubmed_id
        ID, title, abstract, document = AI.parse_document(document_ID)
        print("OUTPUT: Document")
        app.logger.info("OUTPUT: Document")
        return jsonify(
            pubmed_id=document_ID,
            document=document
        )


@ns.route('/extract-genes/<string:pubmed_id>')
@api.response(404, 'Pubmed ID not found or not valid.')
@api.response(200, 'Name, ID and type of each element')
@api.doc(params={'pubmed_id': {"description": "Pubmed ID of an article", "in": "query", "type": "string", "required": "true", "example": " 32819603"}})
class PubtatorGenes(Resource):
    @token_required
    def get(self, pubmed_id):
        logging.info("Extract genes started")
        """Return list of genes extracted from Pubtator"""
        print('Running genes extraction')
        document_ID = pubmed_id
        gene_df, disease_df = AI.pubtator(document_ID)
        result = gene_df.to_json()
        parsed = json.loads(result)
        print("OUTPUT: Pubtator Genes")
        app.logger.info("OUTPUT: Pubtator Genes")
        print(parsed)
        return parsed


@ns.route('/extract-diseases/<string:pubmed_id>')
@api.response(404, 'Pubmed ID not found or not valid.')
@api.response(200, 'Name, ID and type of each element')
@api.doc(params={'pubmed_id': {"description": "Pubmed ID of an article", "in": "query", "type": "string", "required": "true", "example": " 32819603"}})
class PubtatorDiseases(Resource):
    @token_required
    def get(self, pubmed_id):
        logging.info("Extract diseases started")
        """Return list of diseases extracted from Pubtator"""
        print('Running genes extraction')
        document_ID = pubmed_id
        gene_df, disease_df = AI.pubtator(document_ID)
        result = disease_df.to_json()
        parsed = json.loads(result)
        print("OUTPUT: Pubtator Diseases")
        app.logger.info("OUTPUT: Pubtator Diseases")
        print(parsed)
        return parsed


@ns.route('/extract-associations/<string:pubmed_id>')
@api.response(404, 'Pubmed ID not found or not valid.')
@api.response(200, "Pairwise association between gene and disease (both represented with ID)")
@api.doc(params={'pubmed_id': {"description": "Pubmed ID of an article", "in": "query", "type": "string", "required": "true", "example": " 32819603"}})
class Associations(Resource):
    @token_required
    def get(self, pubmed_id):
        logging.info("Extract associations started")
        """Return gene-disease association extracted from document"""
        print('Running associations extraction')
        document_ID = pubmed_id
        ID, title, abstract, document = AI.parse_document(document_ID)
        gene_df, disease_df = AI.pubtator(document_ID)
        pairs = AI.get_pairs(gene_df, disease_df)
        result = AI.get_associations(document, ID, pairs)
        result = result.to_json()
        parsed = json.loads(result)
        print("OUTPUT: Associations")
        app.logger.info("OUTPUT: Associations")
        print(parsed)
        return parsed


if __name__ == '__main__':
    logging.info("Executing main on app")
    app.run(host='0.0.0.0', port=5000)
