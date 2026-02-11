import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

# import tools
from src.tools.retriever import search_f1_regulations
from src.tools.f1_stats import get_race_results_tool

load_dotenv()

# setup llm
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    temperature=0.2,
)
chat_llm = ChatHuggingFace(llm=llm)

# define router chain
router_template = """
You are an expert F1 Assistant Router. Your job is to classify the user's question into one of three categories:

1. "REGULATIONS": Questions about rules, penalties, technical specs, car legality, or sporting procedures.
2. "RESULTS": Questions about race outcomes, who won, podiums, or driver standings for a specific year.
3. "CHAT": General conversation, history, or questions not covered by the above.

Return ONLY the category name. Do not explain.

Question: {question}
Category:
"""

router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | chat_llm | StrOutputParser()

# def handler function
def route_decision(info):
    category = info["category"].strip()
    question = info["question"]
    print(f"--- ROUTING DECISION: {category} ---")

    if "REGULATIONS" in category:
        # call vectordb tool
        return search_f1_regulations.invoke({"query": question, "year": 2026})
    
    elif "RESULTS" in category:
        return "NEEDS_TOOL_CALL"
    
    else:
        return "GENERAL_CHAT"
    
# final chain (takes context and answers the user)
answer_template = """
You are an F1 Official Assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: 
{question}

Answer:
"""
answer_prompt = ChatPromptTemplate.from_template(answer_template)
final_chain = answer_prompt | chat_llm | StrOutputParser()

# main pipeline
def full_chain(question: str):
    # route
    category = router_chain.invoke({"question": question})

    # get context
    if "REGULATIONS" in category:
        context = search_f1_regulations.invoke({"query": question, "year": 2026})
    elif "RESULTS" in category:
        if "Bahrain" in question:
            context = get_race_results_tool.invoke({"year": 2024, "gp_name": "Bahrain"})
        else:
            context = "I can currently only look up Bahrain 2024 results in this demo mode."
    else:
        context = "No specific context needed. Use your general knowledge."

    # generate answer
    return final_chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    q1 = "What is the penalty for changing a gearbox?"
    print(f"Q: {q1}\nA: {full_chain(q1)}\n")