import os
from dotenv import load_dotenv
import getpass
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
import json
from langchain.schema import Document

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
# with open("./bashar.txt", encoding='utf-8') as f:
#     bashar = f.read()

# Inicializar los embeddings y el chunker
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

# Dividir el archivo en fragmentos de 1000 caracteres y probar cada fragmento
# fragment_size = 1000
# fragments = [bashar[i:i+fragment_size] for i in range(0, len(bashar), fragment_size)]

# for i, fragment in enumerate(fragments):
#     try:
#         # Intenta procesar cada fragmento
#         embedding_test = embeddings.embed_query(fragment)
#         print(f"Fragmento {i} procesado con éxito.")
#     except Exception as e:
#         print(f"Error en el fragmento {i}: {e}")

# Crear los fragmentos semánticos
# semantic_chunks = semantic_chunker.create_documents([bashar])

# print(f"Número total de fragmentos semánticos: {len(semantic_chunks)}")

# Guardar los fragmentos semánticos en un archivo JSON con UTF-8 y ensure_ascii=False
# with open("semantic_chunks.json", "w", encoding='utf-8') as f:
#     json.dump([chunk.dict() for chunk in semantic_chunks], f, ensure_ascii=False, indent=4)

# print("Fragmentos semánticos guardados correctamente en semantic_chunks.json")

# Cargar los fragmentos semánticos desde un archivo JSON
with open("semantic_chunks.json", "r", encoding='utf-8') as f:
    semantic_chunks = [Document(**chunk) for chunk in json.load(f)]
print(f"Fragmentos semánticos cargados desde archivo. Total: {len(semantic_chunks)}")


# Crear el vectorstore
semantic_chunk_vectorstore = FAISS.from_documents(
    semantic_chunks,
    embedding=embeddings
)

# Crear el retriever
semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k": 3})

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

    # answer = chain.invoke({"context": context, "question": query})

    # Mostrar la respuesta
    # print("Respuesta del LLM:")
    # print(answer)

### LCEL RAG Chain

# imprimir el contexto y la pregunta antes de la invocación
# print(f"Contexto antes de la invocación: {context}")
# print(f"Pregunta antes de la invocación: {query}")


# LCEL RAG Chain con corrección para pasar solo la pregunta al retriever
semantic_rag_chain = (
    RunnableMap({
        "context": RunnableLambda(lambda x: semantic_chunk_retriever.invoke(x["question"])),  # Pasa solo la pregunta al retriever
        "question": RunnablePassthrough()  # Pasa la pregunta sin modificar
    })
    | RunnableLambda(lambda x: (print("Paso 1: Datos pasados al prompt (RunnableMap):", x), x)[1])  # Depuración
    | prompt  # Genera el prompt con el contexto y la pregunta
    | RunnableLambda(lambda x: (print("Paso 2: Prompt generado:", x), x)[1])  # Depuración
    | llm  # Llama al modelo LLM (GPT-4 o lo que estés usando)
    | RunnableLambda(lambda x: (print("Paso 3: Respuesta del LLM:", x), x)[1])  # Depuración
    | StrOutputParser()  # Procesa la salida del LLM a un string
)

# Ejecutar la cadena
respuesta_rag_chain = semantic_rag_chain.invoke({
    "context": context,  # El contexto cargado o generado por semantic_chunk_retriever
    "question": query    # La pregunta original
})

# Mostrar la respuesta final
print("Respuesta del RAG Chain:", respuesta_rag_chain)