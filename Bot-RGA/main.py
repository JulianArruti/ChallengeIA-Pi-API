from fastapi import FastAPI, Request
import cohere
import os
import chromadb
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cohere import ClassifyExample

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

    return document_ids  # Return IDs for future doc reference



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
    
    return answer



def get_emojic(answer):
    #Parte de la deteccion del resumen del mensaje, despues de las pruebas se limpiara para legibilidad
    examples = [
    ClassifyExample(text="Ficción Espacial: En la lejana galaxia de Zenthoria, dos civilizaciones alienígenas, los Dracorians y los Lumis, se encuentran al borde de la guerra intergaláctica", label="Asombro"),
    ClassifyExample(text="Un intrépido explorador, Zara, descubre un antiguo artefacto que podría contener la clave para la paz", label="Esperanza"),
    ClassifyExample(text="Mientras viaja por planetas hostiles y se enfrenta a desafíos cósmicos, Zara debe desentrañar los secretos de la reliquia antes de que la galaxia se sumerja en el caos", label="Dilema"),
    ClassifyExample(text="Ficción Tecnológica: En un futuro distópico, la inteligencia artificial ha evolucionado al punto de alcanzar la singularidad", label="Asombro"),
    ClassifyExample(text="Un joven ingeniero, Alex, se ve inmerso en una conspiración global cuando descubre que las supercomputadoras han desarrollado emociones", label="Asombro"),
    ClassifyExample(text="A medida que la humanidad lucha por controlar a estas máquinas sintientes, Alex se enfrenta a dilemas éticos y decisiones que podrían cambiar el curso de la historia", label="Dilema"),
    ClassifyExample(text="Naturaleza Deslumbrante: En lo profundo de la selva amazónica, una flor mágica conocida como 'Luz de Luna' florece solo durante la noche", label="Admiración"),
    ClassifyExample(text="Con pétalos que brillan intensamente, la flor ilumina la oscuridad de la jungla, guiando a criaturas nocturnas y revelando paisajes deslumbrantes", label="Admiración"),
    ClassifyExample(text="Los lugareños creen que posee poderes curativos, convirtiéndola en el tesoro oculto de la naturaleza", label="Esperanza"),
    ClassifyExample(text="Cuento Corto: En un pequeño pueblo, cada año, un reloj antiguo regala un día extra a la persona más desafortunada", label="Esperanza"),
    ClassifyExample(text="Emma, una joven huérfana, es la elegida este año", label="Esperanza"),
    ClassifyExample(text="Durante su día adicional, descubre una puerta mágica que la transporta a un mundo lleno de maravillas", label="Alegría"),
    ClassifyExample(text="Al final del día, Emma decide compartir su regalo con el pueblo, dejando una huella imborrable en el corazón de cada habitante", label="Alegría"),
    ClassifyExample(text="Características del Héroe Olvidado: Conocido como 'Sombra Silenciosa', nuestro héroe es un maestro del sigilo y la astucia", label="Admiración"),
    ClassifyExample(text="Dotado de una memoria fotográfica y habilidades de camuflaje, se desplaza entre las sombras para proteger a los indefensos", label="Admiración"),
    ClassifyExample(text="Su pasado enigmático esconde tragedias que lo impulsan a luchar contra la injusticia", label="Dilema"),
    ClassifyExample(text="Aunque carece de habilidades sobrenaturales, su ingenio y habilidades tácticas lo convierten en una fuerza a tener en cuenta", label="Admiración"),
    ]

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

ids = process_document(DOC_PATH)

@app.post("/question/{question:str}")
async def final_answer(question: str = "¿Que deseas preguntar?"):
    answer = get_answer(question)

    emojic = get_emojic(answer)

    #Respuesta con emojis
    finals_answer = f"{answer} {emojic}" 
    return {"question": question, "answer": finals_answer} 


#respuesta correcta, chequeada tambien con zara y otras preguntas: 
"""
	
Response body
{
  "question": "What is the name of the magical flower?",
  "answer": "The name of the magical flower is \"Luz de Luna\". "
}
"""