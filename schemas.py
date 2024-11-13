from pydantic import BaseModel


class Message(BaseModel):
    """Schema for a message."""

    text: str


class chat_schema(BaseModel):
    """Schema for a chat request."""

    text: str
    session_id: str


class Session(BaseModel):
    """Schema for a session."""

    session_id: str
    email_id: str


class SessionCreate(BaseModel):
    """Schema for creating a session."""

    email_id: str


class SessionResponse(BaseModel):
    """Schema for session response."""

    session_id: str
    email_id: str
    started_at: str
