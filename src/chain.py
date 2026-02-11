import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# import tools
from src.tools.retriever import search_f1_regulations
from src.tools.f1_stats import get_race_results_tool
from src.models import Race, Regulations, RouteQuery

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
structured_llm = chat_llm.with_structured_output(RouteQuery)

# define router chain
system_prompt = """You are an expert F1 Assistant. 
Your task is to route the user's question and extract relevant parameters.
- For Rules/Regs (penalties, technical specs): Set intent='REGULATIONS' and fill 'regulations_query'. Default year is 2026.
- For Race Results (winners, podiums): Set intent='RACE_RESULTS' and fill 'race_query'.
- For anything else: Set intent='GENERAL_CHAT'.
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

router_chain = route_prompt | structured_llm
    
# final chain (takes context and answers the user)
answer_prompt = ChatPromptTemplate.from_template("""
You are an F1 Official Assistant. Answer the question based ONLY on the provided context.
If the context is empty or irrelevant, say you don't know.

Context:
{context}

Question: 
{question}

Answer:
""")

final_chain = answer_prompt | chat_llm | StrOutputParser()

# main pipeline
def get_answer(question: str):
    print(f"\n--- Processing: {question} ---")

    # route and extract
    try:
        route_result: RouteQuery = router_chain.invoke({"question": question})
        print(f"--- Detected Intent: {route_result.intent} ---")
    except Exception as e:
        print(f"Routing Error: {e}")
        return "Sorry, I had trouble understanding the intent."

    context = ""

    # invoke tools based on intent
    if route_result.intent == "REGULATIONS":
        # extract params from pydantic object
        year = route_result.requlation_query.year if route_result.requlation_query else 2026
        print(f"--- Tool Call: Searching Regulations ({year}) ---")
        # invoke retriever
        context = search_f1_regulations.invoke({"query": question, "year": year})
    
    elif route_result.intent == "RECE_RESULTS":
        if route_result.race_query:
            year = route_result.race_query.year
            gp_name = route_result.race_query.gp_name
            print(f"--- Tool Call: FastF1 Results ({year} {gp_name}) ---")
            # invoke fastf1 tool
            context = get_race_results_tool.invoke({"year": year, "gp_name": gp_name})
        else:
            context = "Error: Could not extract Race Name or Year."

    else:
        context = "No specific tool needed. Use general knowledge."

    # generate final answer
    print(f"--- Context Length: {len(str(context))} chars ---")
    return final_chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    print(get_answer("What is the maximum fuel mass flow for 2026?"))
    print(get_answer("Who won the Bahrain GP in 2024?"))