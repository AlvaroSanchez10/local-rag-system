import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "technical_docs"

EMBEDDING_FUNCTION = DefaultEmbeddingFunction()


def main():
    client = chromadb.PersistentClient(
        path=VECTORSTORE_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FUNCTION,
    )

    query = "¿Qué tecnología sirve para cachear información y reducir consultas a la base de datos?"

    results = collection.query(
        query_texts=[query],
        n_results=3,
    )

    print("\n=== CONSULTA ===\n")
    print(query)

    print("\n=== RESULTADOS ===\n")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"Resultado {index}")
        print(f"Fuente: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"Distancia: {distance}")
        print(document)
        print("-" * 60)


if __name__ == "__main__":
    main()