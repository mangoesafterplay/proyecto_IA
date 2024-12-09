import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
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
import numpy as np

# Configuración inicial
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("No se encontró la clave GEMINI_API_KEY en el archivo .env")

palm.configure(api_key=gemini_api_key)

# Modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Función para cargar datos del CSV
def load_faq_data(file_path):
    df = pd.read_csv(file_path)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("El archivo CSV debe tener las columnas 'question' y 'answer'")
    return df

faq_path = "faq.csv"
if not os.path.exists(faq_path):
    raise FileNotFoundError(f"No se encontró el archivo '{faq_path}'")
faq_data = load_faq_data(faq_path)
questions = faq_data["question"].tolist()
answers = faq_data["answer"].tolist()

# Crear índice FAISS
embeddings = model.encode(questions)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Funciones auxiliares
def find_closest_question(user_question):
    user_embedding = model.encode([user_question])
    distances, indices = index.search(user_embedding, 1)
    closest_idx = indices[0][0]
    return questions[closest_idx], distances[0][0]

def get_gemini_response(question, context):
    prompt = (
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Por favor responde de manera clara y concisa:"
    )
    model = palm.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text if response else "No se pudo generar una respuesta."

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--psm 6 -l eng+spa'
    extracted_text = pytesseract.image_to_string(binary, config=custom_config)
    try:
        detected_language = detect(extracted_text)
        if detected_language == "en":
            translator = Translator()
            translated_text = translator.translate(extracted_text, src="en", dest="es").text
            return f"Texto extraído: {extracted_text}\nTraducción: {translated_text}"
        elif detected_language == "es":
            return f"Texto extraído: {extracted_text}"
        else:
            return f"Texto extraído (Idioma no identificado): {extracted_text}"
    except Exception as e:
        return f"Error al detectar o traducir el idioma: {e}"


# Interfaz gráfica mejorada
def handle_question():
    user_question = question_entry.get()
    if not user_question:
        messagebox.showerror("Error", "Por favor, ingresa una pregunta.")
        return

    closest_question, distance = find_closest_question(user_question)
    context = f"Pregunta: {closest_question}\nRespuesta: {faq_data.iloc[questions.index(closest_question)]['answer']}"
    response = get_gemini_response(user_question, context)
    output_text.set(f"\n• Pregunta más cercana:\n{closest_question}\n\n• Respuesta:\n{response}")

def handle_image():
    image_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
    if not image_path:
        return
    result = process_image(image_path)
    output_text.delete("1.0", tk.END)  # Limpia el contenido anterior del cuadro de texto
    output_text.insert(tk.END, f"\n• Resultado de la imagen procesada:\n{result}")  # Inserta el nuevo texto


def exit_app():
    root.quit()

# Crear ventana principal
root = tk.Tk()
root.title("Asistente de Aprendizaje de Inglés")
root.geometry("900x650")
root.configure(bg="#1e3d59")

# Estilo
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Times New Roman", 12), background="#1e3d59", foreground="#ffffff")
style.configure("TButton", font=("Times New Roman", 10), padding=6, background="#4caf50", foreground="#ffffff")
style.map("TButton", background=[("active", "#45a049")])

# Título
title_label = ttk.Label(root, text="Bienvenido al Asistente de Inglés", font=("Times New Roman", 18, "bold"), background="#1e3d59", foreground="#ffffff")
title_label.pack(pady=20)

frame = tk.Frame(root, bg="#1e3d59")
frame.pack(pady=20)

question_label = ttk.Label(frame, text="Escribe tu pregunta:", background="#1e3d59")
question_label.grid(row=0, column=0, padx=10, pady=10)

question_entry = ttk.Entry(frame, width=60)
question_entry.grid(row=0, column=1, padx=10, pady=10)

ask_button = ttk.Button(frame, text="Preguntar", command=handle_question)
ask_button.grid(row=0, column=2, padx=10, pady=10)

image_button = ttk.Button(root, text="Cargar Imagen", command=handle_image)
image_button.pack(pady=20)

output_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="solid")
output_frame.pack(pady=20, padx=20, fill="both", expand=True)

scrollbar = tk.Scrollbar(output_frame, orient="vertical")
scrollbar.pack(side="right", fill="y")

output_text = tk.Text(output_frame, wrap="word", font=("Times New Roman", 12), bg="#f7f7f7", fg="#333333", yscrollcommand=scrollbar.set)
output_text.pack(fill="both", expand=True)
scrollbar.config(command=output_text.yview)

# Footer
footer_label = ttk.Label(root, text="Desarrollado por nosotros", font=("Times New Roman", 10), background="#1e3d59", foreground="#ffffff")
footer_label.pack(pady=10)

# Actualización de contenido
def handle_question():
    user_question = question_entry.get()
    if not user_question:
        messagebox.showerror("Error", "Por favor, ingresa una pregunta.")
        return

    closest_question, distance = find_closest_question(user_question)
    context = f"Pregunta: {closest_question}\nRespuesta: {faq_data.iloc[questions.index(closest_question)]['answer']}"
    response = get_gemini_response(user_question, context)
    output_text.delete("1.0", tk.END)  # Limpia el contenido anterior
    output_text.insert(tk.END, f"• Pregunta más cercana:\n{closest_question}\n\n• Respuesta:\n{response}")

def handle_image():
    image_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
    if not image_path:
        return
    result = process_image(image_path)
    output_text.delete("1.0", tk.END)  # Limpia el contenido anterior del cuadro de texto
    output_text.insert(tk.END, f"\n• Resultado de la imagen procesada:\n{result}")  # Inserta el nuevo texto


def exit_app():
    root.quit()

# Iniciar aplicación
root.mainloop()
