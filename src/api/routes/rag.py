from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.serving.config import API_KEY
from src.rag.config import TOP_K
from src.rag.rag import ask


router = APIRouter(
    prefix="/chat",
    tags=["RAG"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K, gt=0, le=20)


class ChatResponse(BaseModel):
    answer: str


@router.post("", response_model=ChatResponse)
def chat(
    data: ChatRequest,
    x_api_key: str = Header(default=None),
):
    # Protect the RAG endpoint using the same API key
    # as the existing prediction endpoint.
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

    try:
        answer = ask(
            question=data.question,
            top_k=data.top_k,
        )

        return ChatResponse(
            answer=answer,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="RAG request failed",
        )