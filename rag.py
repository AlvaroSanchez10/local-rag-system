import asyncio
import logging
import os

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from schemas import RAGResponse


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "technical_docs"
TOP_K = 3

EMBEDDING_FUNCTION = DefaultEmbeddingFunction()


parser = PydanticOutputParser(
    pydantic_object=RAGResponse
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Sos un asistente técnico.

Debes responder exclusivamente usando el CONTEXTO proporcionado.

Reglas:
- No utilices conocimiento externo.
- Si la respuesta no está en el contexto, responde exactamente: "No lo sé".
- No inventes datos.
- No inventes fuentes.
- Incluí únicamente las fuentes realmente utilizadas.

{format_instructions}
""",
        ),
        (
            "human",
            """
CONTEXTO:

{context}

PREGUNTA:

{question}
""",
        ),
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)


model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)


chain = (
    prompt
    | model
    | parser
).with_retry(
    stop_after_attempt=2
)


def get_collection():
    client = chromadb.PersistentClient(
        path=VECTORSTORE_DIR
    )

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FUNCTION,
    )


def retrieve_context(
    query: str,
    top_k: int = TOP_K,
) -> tuple[str, list[str]]:

    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    sources = []

    for document, metadata in zip(
        documents,
        metadatas,
    ):
        source = metadata["source"]

        context_parts.append(
            f"Fuente: {source}\n{document}"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(
        context_parts
    )

    return context, sources


async def get_rag_response(
    query: str,
) -> RAGResponse:

    logger.info(
        "Buscando contexto relevante para: %s",
        query,
    )

    context, retrieved_sources = await asyncio.to_thread(
        retrieve_context,
        query,
    )

    logger.info(
        "Fuentes recuperadas: %s",
        retrieved_sources,
    )

    result = await chain.ainvoke(
        {
            "context": context,
            "question": query,
        }
    )

    logger.info(
        "Respuesta generada y validada correctamente."
    )

    return result