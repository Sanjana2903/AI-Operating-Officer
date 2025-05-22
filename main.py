import os
import json
import time
import re
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

from utils.react_validator import validate_trace_output
from agents.ceo_agent import ceo_agent
from agents.cto_agent import cto_agent
from agents.product_agent import product_agent
from tools.gitlab import create_poc_repo_gitlab_fallback
from tools.jira_fallback import create_core_jira_task_with_fallback
from tools.calender_fallback import book_modern_core_clinic_with_fallback
from feedback import auto_score_with_ragas

load_dotenv()
os.makedirs("logs/reasoning", exist_ok=True)

llm = ChatOllama(model="llama3", temperature=0.2)
embedding = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(persist_directory="db", embedding_function=embedding)
retriever = vectorstore.as_retriever()

def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

def generate_paraphrased_answer(docs, query, role):
    top_chunks = docs[:4]
    numbered_quotes = []
    source_map = {}
    idx = 1

    for doc in top_chunks:
        source = doc.metadata.get("source", "unknown")
        url = doc.metadata.get("url", f"/docs/{source}")
        for sent in split_into_sentences(doc.page_content):
            sent = sent.strip().strip('"')
            if len(sent.split()) > 3:
                numbered_quotes.append(f"{idx}. \"{sent}\" — source: {source}")
                source_map[idx] = {"name": source, "url": url}
                idx += 1
            if idx > 4:
                break
        if idx > 4:
            break

    quote_text = "\n".join(numbered_quotes)
    prompt = (
        f"You are a technical writer summarizing direct quotes from {role.title()}-level sources.\n\n"
        f"Write a single paragraph that summarizes the key recommendation about \"{query}\".\n"
        f"• Use lifted quotes for insights and cite them inline using [1], [2] style footnotes.\n"
        f"• You may add one or two generated (original) insights, but they should not be footnoted.\n"
        f"• If an idea is supported by more than one quote, combine footnotes like [1][3].\n"
        f"• Do NOT invent quotes or citations.\n\n"
        f"Quotes:\n{quote_text}"
    )

    answer = llm.invoke(prompt).content.strip()

    # Find used footnote numbers like [1], [2]
    used_ids = sorted(set(map(int, re.findall(r"\[(\d+)\]", answer))))
    citations = [
        f"[{i}. {source_map[i]['name']}]({source_map[i]['url']})"
        for i in used_ids if i in source_map
    ]

    return {
        "answer": answer,
        "paraphrased_blocks": [{"type": "generated", "text": answer}],
        "citations": citations
    }


def generate_suggested_actions(query, summary):
    prompt = (
        f"Based on the following summary of a user's query: \"{query}\", suggest 2–3 tactical next steps the product/infra team can take.\n"
        f"Summary: {summary}\n\nFormat each action as: ▪ [Action]"
    )
    response = llm.invoke(prompt).content.strip()
    return [line.strip() for line in response.split('\n') if line.strip().startswith("▪")]

def generate_agent_reasoning(num_chunks, similarity, persona, tools_used, action_count):
    trace = [
        f"• Retrieved {num_chunks} passages with cosine {similarity:.2f} from {persona} persona sources → high topical overlap"
    ]
    if tools_used and tools_used != "None":
        trace.append(f"• Chose tools: {tools_used} based on REACT steps executed")
        trace.append(f"• Triggered {action_count} automation attempts → see Suggested Actions for outcomes")
    else:
        trace.append("• No tools triggered based on REACT steps")
    return trace

def create_poc_repo_from_prompt(prompt: str) -> str:
    try:
        from tools.github import create_poc_repo_from_prompt as github
        return github(prompt)
    except:
        return create_poc_repo_gitlab_fallback(prompt)

def create_core_jira_task_from_prompt(prompt: str) -> str:
    return create_core_jira_task_with_fallback(prompt)

def schedule_meeting_from_prompt(prompt: str) -> str:
    return book_modern_core_clinic_with_fallback(prompt)

tools = [
    Tool(name="Create GitHub PoC Repo", func=create_poc_repo_from_prompt, description="Create a private GitHub/GitLab PoC repo."),
    Tool(name="Schedule a Meeting", func=schedule_meeting_from_prompt, description="Schedule meeting with infra team."),
    Tool(name="Create JIRA Task", func=create_core_jira_task_from_prompt, description="Create JIRA task with fallback support.")
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
    early_stopping_method="generate"
)

def get_persona_prompt(role):
    if role.lower() == "ceo":
        return ceo_agent()
    elif role.lower() == "cto":
        return cto_agent()
    elif role.lower() in ["product", "evp"]:
        return product_agent()
    else:
        raise ValueError("Unknown role.")
    
def save_reasoning_json(query, role, context_chunks, trace, answer, actions, reasoning, score, latency, hallucination, feedback):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    format_check = validate_trace_output(answer)

    log = {
        "timestamp": timestamp,
        "persona": role,
        "question": query,
        "final_answer": answer,
        "retrieved_chunks": context_chunks,
        "tools_used": reasoning.get("tools", ""),
        "actions_taken": actions,
        "action_results": actions,
        "score": score,
        "p95_latency_ms": latency,
        "hallucination_rate": hallucination,
        "user_feedback": feedback,
        "reasoning_notes": reasoning,
        "format_issues": format_check["issues"],
        "lifted_quote_present": format_check["lifted_present"],
        "generated_quote_present": format_check["generated_present"],
        "react_steps_count": format_check["react_steps"],
        "raw_trace": [
            {
                "thought": step[0].log if hasattr(step[0], "log") else None,
                "action": step[0].tool if hasattr(step[0], "tool") else None,
                "observation": step[1] if isinstance(step, tuple) else None
            }
            for step in trace.get("intermediate_steps", [])
        ]
    }
    with open(f"logs/reasoning/reasoning_{timestamp}.json", "w") as f:
        json.dump(log, f, indent=2)


def ask_question(query: str, role: str):
    docs = retriever.invoke(query, filter={"persona": role.upper()})
    context_chunks = [doc.page_content for doc in docs]

    query_embedding = embedding.embed_query(query)
    doc_embeddings = [embedding.embed_query(doc.page_content) for doc in docs]
    similarity = float(np.mean(cosine_similarity([query_embedding], doc_embeddings)[0])) if doc_embeddings else 0.0

    summary_out = generate_paraphrased_answer(docs, query, role)
    actions = generate_suggested_actions(query, summary_out["answer"])

    trace = agent_executor.invoke({"input": query})
    steps = trace.get("intermediate_steps", [])
    tool_set = set()
    action_lines = []

    for step in steps:
        if isinstance(step, tuple) and hasattr(step[0], "tool"):
            tool_name = step[0].tool
            observation = step[1]
            tool_set.add(tool_name)
            status = "✅ Success"
            if "fail" in observation.lower() or "❌" in observation.lower():
                status = "❌ Failed"
            action_lines.append(f"▪ {status} — {tool_name} → {observation}")

    tools_used = ", ".join(tool_set) if tool_set else "None"
    reasoning = {
        "chunks": len(context_chunks),
        "similarity": f"{similarity:.2f} (cosine)",
        "tools": tools_used,
        "citations": summary_out["citations"],
        "explanation": generate_agent_reasoning(len(context_chunks), similarity, role.title(), tools_used, len(action_lines))
    }

    score = auto_score_with_ragas(query, "\n\n".join(context_chunks), summary_out["answer"])
    hallucination = 3.0

    return {
        "answer": summary_out["answer"],
        "paraphrased_blocks": summary_out["paraphrased_blocks"],
        "actions": action_lines + actions,
        "reasoning": reasoning,
        "score": score,
        "latency": 0.0,
        "hallucination": hallucination,
        "citations": summary_out["citations"],
        "trace": trace,
        "context_chunks": context_chunks,
        "query": query,
        "role": role
    }

# Action endpoints for Streamlit buttons
def create_github_repo(topic: str) -> str:
    return create_poc_repo_from_prompt(topic)

def create_jira_ticket(topic: str) -> str:
    return create_core_jira_task_from_prompt(topic)

def schedule_calendar_meeting(topic: str) -> str:
    return schedule_meeting_from_prompt(topic)

if __name__ == "__main__":
    print("🤖 Ask the AI Operating Officer (Satya / Kevin / Pavan)")
    role = input("Choose persona (CEO / CTO / Product): ").strip()
    query = input("Ask your question: ").strip()
    ask_question(query, role)
