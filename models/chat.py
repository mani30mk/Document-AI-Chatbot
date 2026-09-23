from typing import List, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    session_id: str
    question: str


class SummarizeRequest(BaseModel):
    session_id: str
    filename: str
    text: Optional[str] = None


class YouTubeVideo(BaseModel):
    id: str
    title: str
    channel: str
    duration: str
    url: str
    thumbnail: str


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources_used: int
    youtube_sources: List[YouTubeVideo] = []
