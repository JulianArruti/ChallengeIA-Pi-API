from fastapi import FastAPI, Request
import cohere

app = FastAPI()

@app.get("/")
async def root():
    try:
        co = cohere.Client("YOUR_API_KEY")  # Replace with your actual API key
        print("Cohere client initialized successfully!")
    except Exception as e:
        print(f"Error initializing Cohere client: {e}")
    return {"message": "Hello, world!"}

