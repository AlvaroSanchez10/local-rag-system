# local-rag-system
sistema local usando chromaDB, langChain y async RAG

Pre-entrega 3 del curso de AI Engineering.

Este proyecto implementa un sistema local de recuperación aumentada por generación (RAG) utilizando ChromaDB, LangChain, embeddings, LCEL y Pydantic.

## Objetivo

El sistema recibe una consulta del usuario, recupera los fragmentos más relevantes desde una base vectorial local y genera una respuesta utilizando exclusivamente el contexto recuperado.

Si la respuesta no se encuentra en el contexto, el sistema debe responder:

```text
No lo sé