import asyncio

from rag import get_rag_response


async def main():

    pregunta_valida = (
        "¿Qué tecnología se utiliza como caché "
        "para reducir consultas a la base de datos?"
    )

    pregunta_trampa = (
        "¿Quién ganó el Mundial de fútbol de 2022?"
    )

    print("\n=== PRUEBA 1: PREGUNTA CON RESPUESTA EN EL CONTEXTO ===\n")

    respuesta_valida = await get_rag_response(
        pregunta_valida
    )

    print(
        respuesta_valida.model_dump_json(
            indent=2
        )
    )

    print("\n=== PRUEBA 2: PREGUNTA FUERA DE CONTEXTO ===\n")

    respuesta_trampa = await get_rag_response(
        pregunta_trampa
    )

    print(
        respuesta_trampa.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())