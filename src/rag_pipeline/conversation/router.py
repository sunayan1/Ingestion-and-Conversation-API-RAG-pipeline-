import uuid
from fastapi import APIRouter

from rag_pipeline.conversation.schemas import ChatRequest, ChatResponse
from rag_pipeline.conversation.memory import get_history, save_history
from rag_pipeline.conversation.rag import get_rag_response
from rag_pipeline.conversation.booking import check_booking_intent_and_extact
from rag_pipeline.db.session import save_booking

router = APIRouter()

# @router.post("/", response_model=ChatResponse)
# async def chat(request: ChatRequest) -> ChatResponse: 
#     session_id = request.session_id or str(uuid.uuid4())

#     history = get_history(session_id)

#     answer = get_rag_response(session_id, request.message, history)
#     history.append({"role": "user", "content": request.message})
#     history.append({"role": "assistant", "content": answer})
#     save_history(session_id, history)

#     return ChatResponse(session_id=session_id, answer=answer)

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    history = get_history(session_id)

    booking_info = check_booking_intent_and_extact(request.message)

    if booking_info:
        save_booking(
            session_id=session_id,
            name=booking_info["name"],
            email=booking_info["email"],
            interview_date=booking_info["date"],
            interview_time=booking_info["time"],
        )
        answer = (
            f"Great, I've booked your interview for {booking_info['date']} at "
            f"{booking_info['time']}. A confirmation will be sent to {booking_info['email']}."
        )
    else:
        answer = get_rag_response(session_id, request.message, history)

    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})
    save_history(session_id, history)

    return ChatResponse(session_id=session_id, answer=answer)