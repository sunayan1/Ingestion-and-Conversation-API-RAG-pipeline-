from datetime import datetime
from sqlalchemy import create_engine, String, Integer, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

engine = create_engine("sqlite:///documents.db", echo=True)

class Base(DeclarativeBase):
    pass

class Documents(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    chunking_strategy: Mapped[str] = mapped_column(String)
    chunk_count: Mapped[int] = mapped_column(Integer)

class Bookings(Base):
    __tablename__ = "bookings"

    booking_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String)
    interview_date: Mapped[str] = mapped_column(String)
    interview_time: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

Base.metadata.create_all(engine)