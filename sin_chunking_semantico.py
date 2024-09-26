import os
from dotenv import load_dotenv
import getpass
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave de API de OpenAI de las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    # Si no se encuentra la clave, solicitarla al usuario de forma segura
    api_key = getpass.getpass("Introduce tu clave de API de OpenAI: ")

os.environ["OPENAI_API_KEY"] = api_key

# Definir la pregunta
query = "¿Somos la única civilización con la que interactúa Bashar?"

# Definir el prompt sin contexto
template = """
Consulta del usuario:
{question}
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Inicializar el LLM
llm = ChatOpenAI(model="gpt-4")

# Crear la cadena
chain = prompt | llm

# Ejecutar la cadena
answer = chain.invoke({"question": query})

# Mostrar la respuesta
print("Respuesta del LLM (sin embeddings):")
print(answer)