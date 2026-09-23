from typing import List, Optional
from pydantic import BaseModel


class NewSessionRequest(BaseModel):
    device_id: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    updated_at: str


class SessionResponse(BaseModel):
    session_id: str
    files: List[str]
