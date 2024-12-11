import os
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
import tkinter as tk
from tkinter import scrolledtext

# Cargar la clave de la API de OpenAI desde el archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configurar LangChain con ChatOpenAI
langchain_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai.api_key
)

# Crear una instancia de ConversationChain para manejar la conversación
conversation_chain = ConversationChain(llm=langchain_llm)

# Crear embeddings para preguntas frecuentes
model = SentenceTransformer('all-MiniLM-L6-v2')

# Función para cargar el archivo FAQ CSV
def load_faq_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    df = pd.read_csv(file_path)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("El archivo CSV debe tener las columnas 'question' y 'answer'")
    return df

# Crear embeddings de las preguntas en el CSV
def create_faq_embeddings(df):
    questions = df["question"].tolist()
    embeddings = model.encode(questions, convert_to_tensor=True)
    return embeddings

# Función para encontrar la pregunta más similar en el CSV
def find_most_similar_question(question, faq_data, faq_embeddings):
    question_embedding = model.encode([question])
    similarities = util.pytorch_cos_sim(question_embedding, faq_embeddings)[0]
    most_similar_idx = similarities.argmax().item()
    return faq_data.iloc[most_similar_idx]

# Función para manejar interacción usando LangChain con el FAQ
def langchain_interact_with_faq(question, faq_data, faq_embeddings):
    # Buscar la pregunta más similar en el FAQ
    most_similar = find_most_similar_question(question, faq_data, faq_embeddings)
    faq_answer = most_similar["answer"]
    
    # Hacer que LangChain genere una respuesta usando la respuesta del FAQ
    prompt = f"Pregunta: {question}\nRespuesta del FAQ: {faq_answer}\nResponde de forma clara y amigable:"
    response = conversation_chain.run(prompt)
    return response

# Función para manejar la interfaz gráfica
def chat_interface():
    # Cargar datos y embeddings
    faq_data = load_faq_data('faq.csv')
    faq_embeddings = create_faq_embeddings(faq_data)

    # Crear ventana principal
    window = tk.Tk()
    window.title("Asistente de FAQ con LangChain")
    window.geometry("800x700")
    window.configure(bg="#edf2f7")

    # Fondo degradado
    def create_gradient(canvas, width, height):
        for i in range(height):
            color = f"#{int(255 - (i / height) * 50):02x}{int(245 - (i / height) * 50):02x}{255:02x}"
            canvas.create_line(0, i, width, i, fill=color)
    
    canvas = tk.Canvas(window, width=800, height=700)
    canvas.pack(fill="both", expand=True)
    create_gradient(canvas, 800, 700)

    # Título
    title = tk.Label(window, text="Asistente de FAQ con LangChain", font=("Helvetica", 24, "bold"), bg="#edf2f7", fg="#2d3748")
    title.place(x=250, y=20)

    # Subtítulo
    subtitle = tk.Label(window, text="Escribe tu pregunta y obtén una respuesta.", font=("Helvetica", 14), bg="#edf2f7", fg="#4a5568")
    subtitle.place(x=220, y=70)

    # Cuadro de historial de chat
    chat_frame = tk.Frame(window, bg="#edf2f7", bd=2, relief="groove")
    chat_frame.place(x=50, y=110, width=700, height=450)

    chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state='normal', bg="#ffffff", fg="#000", font=("Helvetica", 12), bd=0)
    chat_history.config(state='normal')
    chat_history.pack(pady=10, padx=10, fill="both", expand=True)

    # Campo de entrada del usuario
    user_input_frame = tk.Frame(window, bg="#edf2f7")
    user_input_frame.place(x=50, y=580, width=700)

    user_input = tk.Entry(user_input_frame, font=("Helvetica", 14), width=50, bd=2, relief="groove")
    user_input.grid(row=0, column=0, padx=(20, 10), pady=10)  # Añadido padx=(20, 10)

    # Función para manejar el envío del mensaje
    def send_message():
        user_message = user_input.get().strip()  # Obtener texto del usuario
        if not user_message:
            return  # No hacer nada si el usuario no ingresa texto

        # Mostrar mensaje del usuario en el cuadro de texto
        chat_history.config(state='normal')
        chat_history.insert(tk.END, f"Tú: {user_message}\n", "user")

        # Respuesta con LangChain usando el FAQ
        langchain_response = langchain_interact_with_faq(user_message, faq_data, faq_embeddings)
        chat_history.insert(tk.END, f"Asistente: {langchain_response}\n", "langchain")

        # Limpiar el campo de entrada del usuario
        user_input.delete(0, tk.END)

        # Desplazarse al final del historial
        chat_history.see(tk.END)
        chat_history.config(state='disabled')

    # Botón de enviar
    send_button = tk.Button(user_input_frame, text="Enviar", command=send_message, font=("Helvetica", 14), bg="#4CAF50", fg="#fff", relief="flat", activebackground="#45a049")
    send_button.grid(row=0, column=1, padx=(10, 20))  # Añadido padx=(10, 20)

    # Estilizar mensajes
    chat_history.tag_config("user", foreground="#3182CE", font=("Helvetica", 12, "italic"))
    chat_history.tag_config("langchain", foreground="#E53E3E", font=("Helvetica", 12, "italic"))

    window.mainloop()

if __name__ == "__main__":
    chat_interface()
