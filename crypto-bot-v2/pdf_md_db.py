from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from dotenv import load_dotenv
import openai
import os
import uuid
import nltk

# NLTK downloads for tokenization and POS tagging
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load environment variables (API keys, etc.)
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Define data path and collection name
DATA_PATH = "./data/books"
QDRANT_COLLECTION = "compliance_dB"

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333", timeout=80)

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_qdrant(chunks)

def load_documents():
    # Load markdown files
    markdown_loader = DirectoryLoader(DATA_PATH, glob="*.md")
    markdown_docs = markdown_loader.load()

    # Load PDF files individually using PyPDFLoader
    pdf_docs = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(DATA_PATH, filename))
            pdf_docs.extend(pdf_loader.load())

    # Combine documents from both loaders
    documents = markdown_docs + pdf_docs
    return documents

def split_text(documents: list[Document]):
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_qdrant(chunks: list[Document]):
    # Initialize OpenAI embeddings generator
    embedding_function = OpenAIEmbeddings()

    # Delete the existing collection if it exists
    if QDRANT_COLLECTION in [col.name for col in qdrant_client.get_collections().collections]:
        qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)

    # Create a new collection in Qdrant
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=len(embedding_function.embed_query("test")), distance="Cosine")  # Assuming vector size is uniform for all embeddings
    )

    # Prepare and upload points (embedding + metadata for each chunk)
    for i, chunk in enumerate(chunks):
        embedding = embedding_function.embed_query(chunk.page_content)  # Generate embeddings for each chunk
        point_id = str(uuid.uuid4())  # Generate unique ID for each chunk
        
        # Create a point structure with the embedding and metadata
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }  # Store chunk content and metadata as payload
        )
        
        # Upload each point to Qdrant
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=[point])
        print(f"Uploaded chunk {i+1}/{len(chunks)} to Qdrant")

    print(f"Saved {len(chunks)} chunks to Qdrant collection '{QDRANT_COLLECTION}'.")

if __name__ == "__main__":
    main()
