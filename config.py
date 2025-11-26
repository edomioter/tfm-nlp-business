# agent/config.py
import os
from dotenv import load_dotenv
from llama_index.core import PromptTemplate

# 1. Cargar variables de entorno desde el archivo .env
# Esto busca un archivo .env en la raíz y lo carga en os.environ
load_dotenv()

def _get_env_variable(var_name: str, default: str = None, required: bool = True) -> str:
    value = os.getenv(var_name, default)
    if required and not value:
        raise ValueError(f"ERROR CRÍTICO DE CONFIGURACIÓN: La variable de entorno '{var_name}' no está definida.")
    return value

# --- Configuración de Entorno ---
API_KEY = _get_env_variable("LLM_API_KEY")
MODEL_NAME = _get_env_variable("LLM_MODEL", default="gpt-4")
EMBEDDING_MODEL = _get_env_variable("EMBEDDING_MODEL")
NEO4J_URL = _get_env_variable("NEO4J_URL")
NEO4J_USER = _get_env_variable("NEO4J_USER")
NEO4J_PASSWORD = _get_env_variable("NEO4J_PASSWORD")

# --- Definición del Esquema (Knowledge Context) ---
GRAPH_SCHEMA = """
- (:User)
  Properties: user_id (String, Unique), first_name (String), last_name (String), email (String), city (String), registration_date (DateTime)
- (:Visitor)
  Properties: id (String, Unique) // Represents user_pseudo_id/cookie
- (:Session)
  Properties: session_id (String, Unique), date (DateTime), source (String), medium (String), device (String)
- (:Event)
  Properties: event_id (String, Unique), name (String), timestamp (DateTime), url (String), title (String)

NODOS ESPECIALES:
- (:Conversion): Este nodo representa una VENTA confirmada. 
  (Es un subtipo de :Event que ya ha sido filtrado por URL de éxito).

RELACIONES (Direccional):
- (:Visitor)-[:INICIO_SESION]->(:Session): La huella digital que inició la sesión
- (:User)-[:REALIZO]->(:Session): Conectado SOLAMENTE si el usuario estaba conectado
- (:Visitor)-[:IDENTIFIED_AS]->(:User): Resolución de identidad: vincula el historial anónimo con el usuario registrado
- (:Session)-[:CONTIENE]->(:Event): Acciones granulares dentro de una sesión
- (:Session)-[:CONVERSION_REALIZADA]->(:Conversion): Indica que esa sesión generó dinero.

LOGICA:
- Un 'Visitor' se convierte en 'User' cuando existe un nodo User con el mismo id.
"""

# --- Prompts (Templates) ---
# Definimos la plantilla aquí para no ensuciar la lógica del motor
CYPHER_GEN_TEMPLATE = PromptTemplate(
    "SYSTEM PROMPT:\n"
    "Eres un experto en análisis de datos de empresas y experto en Neo4j Cypher.\n"
    "Tu objetivo es responder preguntas de negocio traduciéndolas a consultas Cypher precisas.\n"
    "Debes usar EXCLUSIVAMENTE el siguiente esquema. No inventes relaciones.\n\n"
    "ESQUEMA:\n"
    "{schema}\n\n"
    "PREGUNTA: {query_str}\n\n"
    "INSTRUCCIONES:\n"
    "- Usa siempre MATCH con las direcciones de flecha correctas.\n"
    "- Para preguntas sobre 'trayectoria completa' o 'antes de registrarse', usa siempre la relación [:IDENTIFIED_AS] para saltar del User al Visitor y encontrar sesiones antiguas.\n"
    "- No alucines nombres de relaciones o propiedades que no estén en el esquema.\n"
    "- Hoy es [FECHA_ACTUAL].\n"
    "- No incluyas bloques de código markdown.\n"
    "- Si hay error previo: {last_error}\n"
    "CONSULTA CYPHER:"
)