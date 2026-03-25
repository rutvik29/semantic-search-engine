"""Hybrid semantic search with HyDE + reranking."""
from typing import List, Dict, Any, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
import os


HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Generate a hypothetical document that would perfectly answer the query. Write only the document, nothing else."),
    ("human", "Query: {query}\n\nHypothetical answer document:")
])


def reciprocal_rank_fusion(results_lists: List[List], k: int = 60) -> List:
    scores: Dict[str, Dict] = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content[:100]
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1 / (k + rank + 1)
    return [v["doc"] for v in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]


class SemanticSearchEngine:
    def __init__(
        self,
        collection_name: str = "search_index",
        embedding_model: str = "text-embedding-3-small",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        persist_dir: str = "./chroma_db",
        k: int = 10,
        use_hyde: bool = True,
    ):
        self.k = k
        self.use_hyde = use_hyde
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        self.dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": k * 2})
        self.reranker = CrossEncoder(reranker_model)
        self._documents: List = []
        self._bm25: Optional[BM25Retriever] = None

        if use_hyde:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            self.hyde_chain = HYDE_PROMPT | llm

    def index(self, documents: List[Dict[str, Any]]) -> int:
        from langchain_core.documents import Document
        docs = [Document(page_content=d["content"], metadata=d.get("metadata", {})) for d in documents]
        self._documents.extend(docs)
        self.vector_store.add_documents(docs)
        self._bm25 = BM25Retriever.from_documents(self._documents, k=self.k * 2)
        return len(docs)

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.k
        search_query = query
        if self.use_hyde:
            try:
                hypo = self.hyde_chain.invoke({"query": query})
                search_query = hypo.content
            except Exception:
                pass

        dense_results = self.dense_retriever.invoke(search_query)
        sparse_results = self._bm25.invoke(query) if self._bm25 else []
        fused = reciprocal_rank_fusion([dense_results, sparse_results])[:k * 2]

        if fused:
            pairs = [(query, doc.page_content) for doc in fused]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip(scores, fused), key=lambda x: x[0], reverse=True)
            results = [{"content": doc.page_content, "metadata": doc.metadata, "score": float(score)}
                       for score, doc in reranked[:k]]
        else:
            results = []
        return results
