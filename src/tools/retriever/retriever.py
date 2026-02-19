import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from config.config import EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_retriever(year: int, k: int = 5):
    """
    Returns a retriever object specifically for the requested year's collection.
    """
    print(f"[get_retriever] Loading year collection: {year}")
    vector_store = Chroma(
        collection_name=str(year),
        embedding_function=embeddings,
        persist_directory="./data/chromadb"
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_retriever_for_collection(collection_name: str, k: int = 5):
    """
    Returns a retriever for an explicit collection name.
    """
    print(f"[get_retriever_for_collection] Loading collection: {collection_name}")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./data/chromadb"
    )
    return vector_store.as_retriever(search_kwargs={"k": k})

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

