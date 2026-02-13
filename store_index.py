from dotenv import load_dotenv
import os

from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_fastembed_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Check your .env file.")

extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# ✅ FastEmbed (no torch)
embeddings = download_fastembed_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ NEW index name so you don’t collide with the old HF-based one
index_name = "medical-chatbot-bge"

# bge-small-en-v1.5 is commonly 384 dims in FastEmbed implementations
dimension = 384

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Upsert
PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("✅ Indexing done:", index_name)
