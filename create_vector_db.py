from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

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

# Function to create and save the vector database
def create_vector_db():
    # Use the custom SentenceTransformerEmbeddings class
    embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    
    # Load documents from the local directory
    loader = PyPDFDirectoryLoader("./data")  # Make sure your documents are in the 'data' folder
    docs = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])  # You can adjust the number of documents as needed
    
    # Extract document texts
    document_texts = [doc.page_content for doc in final_documents]
    
    # Embed the documents using the SentenceTransformer embeddings
    embedded_docs = embeddings.embed_documents(document_texts)
    
    # Update Document objects with embeddings
    for i, doc in enumerate(final_documents):
        doc.metadata["embedding"] = embedded_docs[i]
    
    # Create and save FAISS vector store from documents
    vector_store = FAISS.from_documents(final_documents, embeddings)
    
    # Save the vector store to a local file (this happens only once)
    vector_store.save_local("vector_store.faiss")

    print("Vector database created and saved as 'vector_store.faiss'.")

# Run the vector database creation
if __name__ == "__main__":
    create_vector_db()
