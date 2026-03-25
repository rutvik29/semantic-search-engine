"""FastAPI search API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.search import SemanticSearchEngine
import os

app = FastAPI(title="Semantic Search API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = SemanticSearchEngine()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_hyde: bool = True


class IndexRequest(BaseModel):
    documents: List[Dict[str, Any]]


@app.post("/search")
async def search(req: SearchRequest):
    results = engine.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results, "total": len(results)}


@app.post("/index")
async def index_docs(req: IndexRequest):
    count = engine.index(req.documents)
    return {"indexed": count, "status": "ok"}


@app.get("/health")
def health(): return {"status": "ok"}
