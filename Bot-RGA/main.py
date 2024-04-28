from fastapi import FastAPI, Request
import cohere
import os
import chromadb
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_PATH = os.path.join(BASE_DIR, 'data', 'documento.txt')
with open (DOC_PATH, 'r', encoding="utf-8") as file:
    document_content = file.read()

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="my_collection", metadata={"hnsw:space": "ip"}  # Adjust space if needed
)

co = cohere.Client("G6m6NXL8hYwNWUiXvtMYxfch6BgQk2RpXvg4uTXS")

# Function to split document content into chunks with embeddings
def process_document(document_path):
    # Read document content
    with open(document_path, "r", encoding="utf-8") as file: ###
        document_content = file.read()  ###

    # Split content using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["."], chunk_size=50, chunk_overlap=20
    )
    docs = text_splitter.create_documents([document_content])

    # Generate embeddings and add them to the collection
    document_ids = []
    for doc in docs:
        uuid_name = str(uuid.uuid1())
        print(f"Adding embedding with ID: {uuid_name}")
        embedding_data = co.embed(
            texts=[doc.page_content], model="embed-multilingual-v3.0", input_type="classification"
        ).embeddings
        collection.add(documents=[doc.page_content], ids=[uuid_name], embeddings=embedding_data)
        document_ids.append(uuid_name)

    return document_ids  # Return IDs for future doc reference

#Testing process_document(document_path)
print(process_document(DOC_PATH))

@app.get("/")
async def root():
    try:
        co = cohere.Client("YOUR_API_KEY")  # Replace with your actual API key
        print("Cohere client initialized successfully!")
    except Exception as e:
        print(f"Error initializing Cohere client: {e}")
    return {"message": "Hello, world!"}
