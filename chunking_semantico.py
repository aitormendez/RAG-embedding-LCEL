import os
from dotenv import load_dotenv
import getpass
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
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

# Si utilizas LangSmith o LangChain, puedes hacer lo mismo
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Leer el archivo de texto
with open("./bashar.txt", encoding='utf-8') as f:
    bashar = f.read()

# Inicializar los embeddings y el chunker
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# Crear los fragmentos semánticos
semantic_chunks = semantic_chunker.create_documents([bashar])

print(f"Número total de fragmentos semánticos: {len(semantic_chunks)}")

# Crear el vectorstore
semantic_chunk_vectorstore = FAISS.from_documents(
    semantic_chunks,
    embedding=embeddings
)

# Crear el retriever
semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k": 1})

# Definir la pregunta
query = "¿Somos la única civilización con la que interactúa Bashar?"

# Recuperar los documentos relevantes
docs = semantic_chunk_retriever.invoke(query)

# Verificar si se recuperaron documentos
if not docs:
    print("No se recuperaron documentos relevantes.")
else:
    # Combinar el contenido de los documentos para el contexto
    context = "\n\n".join([doc.page_content for doc in docs])

    # Definir el prompt
    template = """
    Utilice el siguiente contexto para responder la consulta del usuario. Si no puede responder, responda "No lo sé".

    Consulta del usuario:
    {question}

    Contexto:
    {context}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Inicializar el LLM
    llm = ChatOpenAI(model="gpt-4")

    # Crear la cadena
    chain = prompt | llm

    # Ejecutar la cadena

    answer = chain.invoke({"context": context, "question": query})

    # Mostrar la respuesta
    print("Respuesta del LLM:")
    print(answer)