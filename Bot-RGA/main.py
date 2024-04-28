from fastapi import FastAPI, Request
import cohere
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_PATH = os.path.join(BASE_DIR, 'data', 'documento.txt')

with open (DOC_PATH, 'r', encoding="utf-8") as file:
    document_content = file.read()

print(document_content)

@app.get("/")
async def root():
    try:
        co = cohere.Client("YOUR_API_KEY")  # Replace with your actual API key
        print("Cohere client initialized successfully!")
    except Exception as e:
        print(f"Error initializing Cohere client: {e}")
    return {"message": "Hello, world!"}


