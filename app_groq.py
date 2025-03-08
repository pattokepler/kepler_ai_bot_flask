from flask import Flask, render_template, request, redirect, url_for
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import time
import numpy as np

load_dotenv()

app = Flask(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """You are an HR. You should answer general questions yourself; you should be polite.
    You shouldn't let them know you are a bot.
    Answer the questions on the provided context only. Please provide the most accurate response based on the question.
    <context>{context}</context>
    Questions: {input}
    """
)

# Custom wrapper class for SentenceTransformer to integrate with LangChain's Embeddings
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Embedding the list of texts (documents) using SentenceTransformer
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query):
        # Embedding a single query using SentenceTransformer
        return self.model.encode([query])[0]

# Define a function for handling common questions
def handle_common_questions(question):
    common_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I help you with?",
        # Add more common questions and responses here
    }
    return common_responses.get(question.lower())

# Vector embedding function
def vector_embedding():
    if "vectors" not in app.config:
        # Use the custom SentenceTransformerEmbeddings class
        app.config['embeddings'] = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        
        # Load documents from the local directory
        app.config['loader'] = PyPDFDirectoryLoader("./data")
        app.config['docs'] = app.config['loader'].load()
        
        # Split documents into chunks
        app.config['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        app.config['final_documents'] = app.config['text_splitter'].split_documents(app.config['docs'][:20])
        
        # Create Document objects
        document_texts = [doc.page_content for doc in app.config['final_documents']]
        
        # Embed the documents using the SentenceTransformer embeddings
        embeddings = app.config['embeddings'].embed_documents(document_texts)
        
        # Update Document objects with embeddings using langchain
        for i, doc in enumerate(app.config['final_documents']):
            doc.metadata["embedding"] = embeddings[i]
        
        # Create FAISS vector store from documents
        app.config['vectors'] = FAISS.from_documents(app.config['final_documents'], app.config['embeddings'])

# Format response to handle line breaks
def format_response(response):
    formatted_response = response.replace("\n", "<br>")
    return formatted_response

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    vector_embedding()  # Initialize embeddings

    if 'messages' not in app.config:
        app.config['messages'] = []

    if request.method == 'POST':
        question = request.form.get('question')

        common_response = handle_common_questions(question)
        if common_response:
            response = common_response
        else:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = app.config["vectors"].as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Start timing the response generation
            start = time.process_time()
            response = retrieval_chain.invoke({'input': question})['answer']
            #response_time = time.process_time() - start
            #response += f" (Response Time: {response_time:.2f} seconds)"

        app.config['messages'].append((question, response))

    return render_template('index.html', messages=app.config.get('messages', []))

if __name__ == '__main__':
    app.run(debug=False)
