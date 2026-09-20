from sqlalchemy.orm import sessionmaker
from rag_pipeline.db.models import engine, Documents, Bookings
import uuid

SessionLocal = sessionmaker(bind=engine)

def save_document_metadata(document_id: str, filename: str, chunking_strategy: str, chunk_count: int) -> None: 
    session = SessionLocal()
    try: 
        doc = Documents(
            document_id=document_id,
            filename=filename,
            chunking_strategy=chunking_strategy,
            chunk_count=chunk_count,
        )
        session.add(doc)
        session.commit()
    finally: 
        session.close()

def save_booking(session_id: str, name: str, email: str, interview_date: str, interview_time: str) -> None:
    
    session = SessionLocal()
    try:
        booking = Bookings(
            booking_id=str(uuid.uuid4()),
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )
        session.add(booking)
        session.commit()
    finally:
        session.close()