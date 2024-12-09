import os
from dotenv import load_dotenv
import google.generativeai as palm
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import pytesseract
from googletrans import Translator
import cv2
from langdetect import detect

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Cargar clave de API desde .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("No se encontró la clave GEMINI_API_KEY en el archivo .env")

# Configurar la API de Gemini
palm.configure(api_key=gemini_api_key)

# Modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Función para cargar datos del CSV
def load_faq_data(file_path):
    df = pd.read_csv(file_path)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("El archivo CSV debe tener las columnas 'question' y 'answer'")
    return df

# Crear índice FAISS
def create_faiss_index(questions):
    embeddings = model.encode(questions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Encontrar la pregunta más cercana
def find_closest_question(user_question, index, questions, k=1):
    user_embedding = model.encode([user_question])
    distances, indices = index.search(user_embedding, k)
    closest_idx = indices[0][0]
    return questions[closest_idx], distances[0][0]

# Función para obtener respuesta usando Gemini
def get_gemini_response(question, context):
    # Construimos la consigna que incluye el contexto
    prompt = (
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Por favor responde de manera clara y concisa:"
    )
    
    # Usamos el modelo generativo para crear contenido
    model = palm.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    # Devolvemos el texto generado
    return response.text if response else "No se pudo generar una respuesta."

# Nueva funcionalidad: Procesar imágenes
def process_image(image_path):
    # Verificar si el archivo de imagen existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró el archivo de imagen '{image_path}'.")
    
    print(f"\nProcesando imagen: {image_path}")
    
    # Leer la imagen y convertirla a escala de grises
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar binarización para mejorar el contraste
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extraer texto usando Tesseract con configuración optimizada
    custom_config = r'--psm 6 -l eng+spa'
    extracted_text = pytesseract.image_to_string(binary, config=custom_config)

    # Detectar idioma y traducir si es necesario
    translator = Translator()
    try:
        detected_language = detect(extracted_text)
        if detected_language == "en":
            translated_text = translator.translate(extracted_text, src="en", dest="es").text
            print(f"Texto extraído (Inglés detectado):\n{extracted_text}")
            print(f"Traducción al español:\n{translated_text}")
        elif detected_language == "es":
            print(f"Texto extraído (Español detectado):\n{extracted_text}")
        else:
            print("Texto extraído (Idioma no identificado):")
            print(extracted_text)
    except Exception as e:
        print(f"Error al detectar/traducir el idioma: {e}")

# Programa principal
def main():
    faq_path = "faq.csv"
    if not os.path.exists(faq_path):
        raise FileNotFoundError(f"No se encontró el archivo '{faq_path}'.")
    
    print("Cargando datos de FAQ...")
    faq_data = load_faq_data(faq_path)
    questions = faq_data["question"].tolist()
    answers = faq_data["answer"].tolist()

    print("Creando índice vectorial...")
    index, embeddings = create_faiss_index(questions)

    print("\nEscribe tu pregunta o ruta de imagen (o 'salir' para terminar):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "salir":
            print("¡Hasta luego!")
            break
        
        if user_input.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            try:
                process_image(user_input)
            except Exception as e:
                print(f"Error al procesar la imagen: {e}")
        else:
            closest_question, distance = find_closest_question(user_input, index, questions)
            print(f"\nPregunta más cercana encontrada: {closest_question} (distancia: {distance:.2f})")

            context = f"Pregunta: {closest_question}\nRespuesta: {faq_data.iloc[questions.index(closest_question)]['answer']}"
            response = get_gemini_response(user_input, context)
            print(f"\nRespuesta del asistente: {response}\n")

if __name__ == "__main__":
    main()
