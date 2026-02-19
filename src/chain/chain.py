import os
from operator import itemgetter
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from .chat_history import add_to_history, get_chat_history, get_session_history, get_chat_history_list

# import tools
from src.tools import get_retriever, get_retriever_for_collection, get_session_collection
from src.guardrails.checks import validate_input, validate_output
from config.config import LLM_PROVIDER, LLM_MODEL, GEMINI_KEY, HF_TOKEN, HF_BASE_URL


# LLM Setup
match LLM_PROVIDER:
    case "gemini":
        chat_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            api_key=GEMINI_KEY
        )
    case "huggingface":
        chat_llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=HF_TOKEN,
            base_url=HF_BASE_URL
        )


# Answer Chain Prompt
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an F1 Official Assistant. Answer the question based ONLY on the provided context.
If the context is empty or irrelevant, say you don't know.
Be concise but thorough. Always cite the specific regulation or source when possible.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def fetch_context(input_dict: dict, config: RunnableConfig) -> str:
    """
    Fetch context using session_id from RunnableConfig (set by RunnableWithMessageHistory).
    This is the correct way to access session_id inside a LCEL chain.
    """
    question = input_dict["question"]
    year = input_dict.get("year", 2026)

    # session_id is in config["configurable"]["session_id"], NOT in input_dict
    session_id = config.get("configurable", {}).get("session_id", "default")

    print(f"\n[fetch_context] Session ID: {session_id}")
    collection_name = get_session_collection(session_id)
    print(f"[fetch_context] Uploaded collection: {collection_name}")

    if collection_name:
        print(f"[fetch_context] Using uploaded collection: {collection_name}")
        retriever = get_retriever_for_collection(collection_name)
    else:
        print(f"[fetch_context] No upload found. Using default year collection: {year}")
        retriever = get_retriever(year)

    docs = retriever.invoke(question)
    print(f"[fetch_context] Retrieved {len(docs)} documents")

    return "\n\n".join(
        [f"[Source: {d.metadata.get('source', 'Unknown')} | Rule: {d.metadata.get('rule_id', 'N/A')}]\n{d.page_content}"
         for d in docs]
    )


# base chain
base_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(fetch_context)  # RunnableLambda passes config automatically
    )
    | answer_prompt
    | chat_llm
    | StrOutputParser()
    | RunnableLambda(lambda answer: {"answer": answer})
)

# RAG chain with history
rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# ============ Main Pipeline ============
def get_answer(question: str, session_id: str = "default", year: int = 2026) -> dict:
    """
    RAG pipeline with vector database retrieval, guardrails, and chat history.
    """
    print(f"\n--- Processing: {question} ---")

    # Input validation (guardrails)
    is_valid, validation_msg = validate_input(question)
    if not is_valid:
        return {
            "answer": f"I cannot process this request: {validation_msg}",
            "sources": [],
            "validation_info": {"input_blocked": True, "reason": validation_msg}
        }

    print(f"\n--- Retrieving documents from vector database ({year}) ---")
    print(f"[get_answer] Session ID: {session_id}")
    collection_name = get_session_collection(session_id)
    print(f"[get_answer] Uploaded collection: {collection_name}")

    if collection_name:
        print(f"[get_answer] Using uploaded collection: {collection_name}")
        retriever = get_retriever_for_collection(collection_name)
    else:
        print(f"[get_answer] No upload found. Using default year collection: {year}")
        retriever = get_retriever(year)

    retrieved_docs = retriever.invoke(question)
    print(f"[get_answer] Retrieved {len(retrieved_docs)} documents")

    context = "\n\n".join(
        [f"[Source: {d.metadata.get('source', 'Unknown')} | Rule: {d.metadata.get('rule_id', 'N/A')}]\n{d.page_content}"
         for d in retrieved_docs]
    )

    history = get_chat_history(session_id)

    print(f"--- Context Length: {len(str(context))} chars ---")

    chain_part = answer_prompt | chat_llm | StrOutputParser()
    raw_answer = chain_part.invoke({
        "context": context,
        "question": question,
        "chat_history": history
    })

    final_answer, validation_info = validate_output(raw_answer, retrieved_docs)
    add_to_history(session_id, question, raw_answer)

    return {
        "answer": final_answer,
        "sources": [d.metadata for d in retrieved_docs] if retrieved_docs else [],
        "validation_info": validation_info
    }


# ============ Simple Interface ============
def chat(question: str, session_id: str = "default", year: int = 2026) -> str:
    """Simple chat interface that returns just the answer string."""
    result = get_answer(question, session_id, year)
    return result["answer"]


if __name__ == "__main__":
    print("=" * 60)
    print(chat("What is the maximum engine horse power of an f1 car?", session_id="test"))