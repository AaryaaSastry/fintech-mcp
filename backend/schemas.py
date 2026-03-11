from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str

class QueryRequest(BaseModel):
    client_id: Optional[int] = None
    transaction_id: Optional[int] = None
    mcc_code: Optional[str] = None
    months: int = 3
