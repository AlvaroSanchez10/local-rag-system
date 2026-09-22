import re
from pathlib import Path

import chromadb
import tiktoken
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "technical_docs"

EMBEDDING_FUNCTION = DefaultEmbeddingFunction()


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def get_token_length(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def load_documents() -> list[dict]:
    documents = []

    for file_path in DATA_DIR.glob("*"):
        if file_path.suffix not in {".txt", ".md"}:
            continue

        content = file_path.read_text(
            encoding="utf-8"
        )

        content = clean_text(content)

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


def split_documents(
    documents: list[dict],
) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=get_token_length,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:
        parts = splitter.split_text(
            document["content"]
        )

        for index, part in enumerate(parts):
            chunks.append(
                {
                    "id": f"{document['source']}-{index}",
                    "content": part,
                    "source": document["source"],
                    "chunk_index": index,
                }
            )

    return chunks


def ingest_documents():
    client = chromadb.PersistentClient(
        path=VECTORSTORE_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FUNCTION,
    )

    if collection.count() > 0:
        print(
            "La colección ya contiene documentos. "
            "No se volverá a indexar."
        )
        return

    documents = load_documents()

    chunks = split_documents(
        documents
    )

    if not chunks:
        print(
            "No se encontraron documentos para procesar."
        )
        return

    collection.upsert(
        ids=[
            chunk["id"]
            for chunk in chunks
        ],
        documents=[
            chunk["content"]
            for chunk in chunks
        ],
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    print(
        f"Ingesta completada. "
        f"Se guardaron {len(chunks)} chunks."
    )


if __name__ == "__main__":
    ingest_documents()