from fastapi import FastAPI, Request, Body
import cohere
import os
import chromadb
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cohere import ClassifyExample
from data.sentiment_examples import examples
import re

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_PATH = os.path.join(BASE_DIR, 'data', 'documento.txt')
with open (DOC_PATH, 'r', encoding="utf-8") as file:
    document_content = file.read()

# Iniciando cliente ChromaDB y coleccion
# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="my_collection", metadata={"hnsw:space": "ip"}  # Adjust space if needed
)

#Conexion a cliente
co = cohere.Client("G6m6NXL8hYwNWUiXvtMYxfch6BgQk2RpXvg4uTXS")

# Function para realizar los chunks y agregarlos a coleccion
def process_document(document_path): 
    #Leyendo el documento
    with open(document_path, "r", encoding="utf-8") as file: ###
        document_content = file.read()  ###

    # Funcion para generar el split usando RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["."], chunk_size=50, chunk_overlap=20
    )
    docs = text_splitter.create_documents([document_content])

    # Generando el embedding y agregandolo a coleccion
    document_ids = []
    for doc in docs:
        uuid_name = str(uuid.uuid1())
        print(f"Adding embedding with ID: {uuid_name}")
        embedding_data = co.embed(
            texts=[doc.page_content], model="embed-multilingual-v3.0", input_type="classification"
        ).embeddings
        collection.add(documents=[doc.page_content], ids=[uuid_name], embeddings=embedding_data)
        document_ids.append(uuid_name)

def get_first_sentence(text):
    # Funcion para devolver sola la primera oracion
    match = re.search(r'(?<=[.!?]) +', text)
    if match:
        return text[:match.start()]
    else:
        return text

def get_answer(question):
   #Realizando embedding de la pregunta
    question_embedding = co.embed(
        texts=[question], model="embed-multilingual-v3.0", input_type="classification"
    ).embeddings[0]

    #Obtencion de los documentos mas cercano para responder a la pregunta
    context = collection.query(query_embeddings=[question_embedding], n_results=2)["documents"][0]
    
    #Permitiendo que tome varios documentos para contexto
    document = []
    for doc in context:
        document.append({"snippet": doc})

    #Obteniendo la respuesta del MML
    answer = co.chat(
        message=question,
        documents=document).text
    
    # Extrae la primera oración de la respuesta
    answer2 = get_first_sentence(answer)
    return answer2

def get_emojic(answer):
    #usando co.classify obtendre el resumen del contenido de cada documento en un emoji
    sentiment = co.classify(
        inputs=[answer],
        examples=examples,
        model= 'embed-multilingual-v2.0'
        )
    
    for classification in sentiment.classifications:
        sentiment = (classification.prediction)
 
    sentiment_emojis = {
        "Alegría": "😀",
        "Dilema": "😧",
        "Asombro": "😲",
        "Esperanza": "🙌",
        "Admiracion": "😯"
    }
    # Devuelve el emoji correspondiente al sentimiento
    emojic = sentiment_emojis.get(sentiment, "")
    return emojic

process_document(DOC_PATH)

@app.post("/question/{username}/{question}")
async def final_answer(username: str, question: str = "¿Que deseas preguntar?"):
    answer = get_answer(question)

    emojic = get_emojic(answer)

    #Respuesta con emojis
    finals_answer = f"{answer} {emojic}" 
    return {"username": username, "question": question, "answer": finals_answer} 