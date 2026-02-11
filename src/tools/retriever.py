import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

embeddings = ...

def get_retriever(year: int):
    """
    Returns a retriever object specifically for the requested year's collection.
    """
    vector_store = Chroma(
        collection_name=str(year),
        embedding_function=embeddings,
        presist_directory="./data/chromadb"
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})

@tool
def search_f1_regulations(query: str, year: int=2026):
    """
    Search the Formula 1 regulations (Technical, Sporting, Financial) for a specific year.
    Useful for questions about rules, penalties, car specs, or procedures.
    """
    retriever = get_retriever(year)
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in docs]
        )
    return context

