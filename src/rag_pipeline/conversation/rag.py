from rag_pipeline.conversation.retrieval import retrieve_relevant_chunks
from rag_pipeline.conversation.llm import generate_response

def get_rag_response(session_id:str, question:str, history:list[dict]) -> str: 
    relevant_chunks = retrieve_relevant_chunks(question)
    context = "\n\n".join(relevant_chunks)

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant answering questions based on the provided "
            "document context. If the answer isn't in the context, say you don't know "
            "rather than making something up."
        ),
    }

    user_message_with_context = {
        "role": "user",
        "content": f"Context from documents:\n{context}\n\nQuestion: {question}",
    }

    messages = [system_message] + history + [user_message_with_context]

    answer = generate_response(messages)
    return answer