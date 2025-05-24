import os
import json
import time
import re
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import requests
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
from urllib.parse import quote_plus
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

load_dotenv()

os.makedirs("logs/reasoning", exist_ok=True)

llm = ChatOllama(model="llama3", temperature=0.2)
embedding = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(persist_directory="db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})


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
        f"Write a single paragraph summarizing the key recommendation about '{query}'.\n"
        f"• Use only the lifted quotes from below.\n"
        f"• Cite them inline using [1], [2] footnotes.\n"
        f"• Do NOT invent or add any new facts or opinions.\n"
        f"• If multiple quotes support a point, cite [1][3].\n\n"
        f"Quotes:\n{quote_text}"
    )

    answer = llm.invoke(prompt).content.strip()

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
    
def save_reasoning_json(query, role, context_chunks, trace, answer, actions, reasoning, score, latency, hallucination, feedback, deepeval_passed):
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
        "deepeval_passed": deepeval_passed,
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
        ] if isinstance(trace, dict) and "intermediate_steps" in trace else []

    }

    with open(f"logs/reasoning/reasoning_{timestamp}.json", "w") as f:
        json.dump(log, f, indent=2)





def rephrase_for_github_search(query: str) -> str:
    prompt = (
        f"Rewrite the following question into a short GitHub search query. "
        f"Only return the raw search phrase — no explanation, no markdown, no code block.\n\n"
        f"Example:\n"
        f"Input: How can I build a chatbot using LangChain?\n"
        f"Output: langchain chatbot example\n\n"
        f"Now rewrite:\n"
        f"Input: {query}\n"
        f"Output:"
    )
    response = llm.invoke(prompt).content.strip()
    response = response.replace("`", "").strip()
    return response.splitlines()[0]

def search_github_repositories(query: str) -> list:
    import requests
    from urllib.parse import quote_plus

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AI-Operating-Officer"
    }
    encoded_query = quote_plus(query)
    params = {
        "q": encoded_query,
        "sort": "stars",
        "order": "desc",
        "per_page": 5
    }

    try:
        response = requests.get("https://api.github.com/search/repositories", headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get("items", [])

        if not items:
            return ["❗ No matching GitHub repositories found."]

        return [f"[{repo['full_name']}]({repo['html_url']})" for repo in items]
    except Exception as e:
        return [f"❌ GitHub search failed: {str(e)}"]



def calculate_ragas_metrics_actual(query: str, contexts_list: list[str], answer: str):
    dataset = Dataset.from_dict({
        'question': [query],
        'answer': [answer],
        'contexts': [contexts_list]
    })

    selected_metrics = [faithfulness, answer_relevancy]
    results = evaluate(dataset, metrics=selected_metrics)

    # Handle both list and scalar return types
    faithfulness_score = results['faithfulness'][0] if isinstance(results['faithfulness'], list) else results['faithfulness']
    relevancy_score = results['answer_relevancy'][0] if isinstance(results['answer_relevancy'], list) else results['answer_relevancy']

    final_score = (faithfulness_score + relevancy_score) / 2
    hallucination = 1.0 - faithfulness_score

    return round(final_score, 2), round(hallucination, 2)


# --- Main Question Handler ---
def ask_question(query: str, role: str):
    try:
        start_time = time.time()
        docs = retriever.invoke(query, filter={"persona": role.lower()})
        if not docs:
            docs = retriever.invoke(query)
        context_chunks_content = [doc.page_content for doc in docs]

        if not context_chunks_content:
            return {
                "answer": "No relevant content was found for this query and persona.",
                "paraphrased_blocks": [],
                "actions": [],
                "reasoning": {"explanation": "Retriever returned no chunks."},
                "score": 0.0,
                "latency": -1,
                "hallucination": 1.0,
                "deepeval_passed": False,
                "citations": [],
                "trace": {},
                "context_chunks": [],
                "query": query,
                "role": role
            }

        query_emb = embedding.embed_query(query)
        top_chunk_emb = embedding.embed_documents([context_chunks_content[0]])[0]
        similarity = sum(a * b for a, b in zip(query_emb, top_chunk_emb)) / (
            sum(a * a for a in query_emb) ** 0.5 * sum(b * b for b in top_chunk_emb) ** 0.5
        )

        summary_out = generate_paraphrased_answer(docs, query, role)
        actions_generated = generate_suggested_actions(query, summary_out["answer"])

        ragas_score_f1, hallucination_rate = calculate_ragas_metrics_actual(
            query, context_chunks_content, summary_out["answer"]
        )

        # 🔁 Retry logic begins — fallback if RAGAS < 0.8
        if ragas_score_f1 < 0.8:
            print("🔁 RAGAS score < 0.8 — retrying with expanded retrieval (k=8)")
            fallback_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
            fallback_docs = fallback_retriever.invoke(query)
            fallback_chunks = [doc.page_content for doc in fallback_docs]

            if fallback_chunks:
                fallback_summary = generate_paraphrased_answer(fallback_docs, query, role)
                fallback_score, fallback_hallucination = calculate_ragas_metrics_actual(
                    query, fallback_chunks, fallback_summary["answer"]
                )

                if fallback_score > ragas_score_f1:
                    print(f"✅ Score improved: {ragas_score_f1} → {fallback_score}")
                    summary_out = fallback_summary
                    context_chunks_content = fallback_chunks
                    ragas_score_f1 = fallback_score
                    hallucination_rate = fallback_hallucination
        # 🔁 Retry logic ends

        agent_loader = {
            "ceo": ceo_agent,
            "cto": cto_agent,
            "product": product_agent
        }.get(role.lower(), ceo_agent)

        agent_executor = agent_loader()
        trace = agent_executor.invoke({
            "question": query,
            "context": "\n\n".join(context_chunks_content)
        })
        steps = trace.get("intermediate_steps", []) if isinstance(trace, dict) else []

        tool_set = set()
        action_lines = []
        for step in steps:
            if isinstance(step, tuple) and hasattr(step[0], "tool"):
                tool_name = step[0].tool
                observation = step[1]
                tool_set.add(tool_name)
                status = "✅ Success"
                if isinstance(observation, str) and ("fail" in observation.lower() or "❌" in observation.lower()):
                    status = "❌ Failed"
                action_lines.append(f"▪ {status} — {tool_name} → {observation}")

        tools_used = ", ".join(tool_set) if tool_set else "None"

        test_case = LLMTestCase(
            input=query,
            actual_output=summary_out["answer"],
            expected_output=summary_out["answer"],
            retrieval_context=context_chunks_content
        )

        deepeval_passed = False
        try:
            assert_test(
                test_case,
                metrics=[
                    FaithfulnessMetric(threshold=0.8),
                    AnswerRelevancyMetric(threshold=0.8),
                    ContextualPrecisionMetric(threshold=0.8),
                ]
            )
            deepeval_passed = True
        except Exception as dee:
            print("🔍 DeepEval failed:", dee)

        reasoning = {
            "chunks": len(context_chunks_content),
            "similarity": f"{similarity:.2f} (cosine)",
            "tools": tools_used,
            "citations": summary_out["citations"],
            "ragas_f1": ragas_score_f1,
            "hallucination_rate": hallucination_rate,
            "deepeval_passed": deepeval_passed,
            "explanation": generate_agent_reasoning(
                len(context_chunks_content), similarity, role.title(), tools_used, len(action_lines)
            )
        }

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return {
            "answer": summary_out["answer"],
            "paraphrased_blocks": summary_out["paraphrased_blocks"],
            "actions": action_lines + actions_generated,
            "reasoning": reasoning,
            "score": ragas_score_f1,
            "latency": latency_ms,
            "hallucination": hallucination_rate,
            "deepeval_passed": deepeval_passed,
            "citations": summary_out["citations"],
            "trace": trace,
            "context_chunks": context_chunks_content,
            "query": query,
            "role": role
        }

    except Exception as e:
        print(f"❌ ask_question failed: {e}")
        return {
            "answer": "An error occurred during processing.",
            "paraphrased_blocks": [],
            "actions": [],
            "reasoning": {"explanation": str(e)},
            "score": 0.0,
            "latency": -1,
            "hallucination": 1.0,
            "deepeval_passed": False,
            "citations": [],
            "trace": {},
            "context_chunks": [],
            "query": query,
            "role": role
        }

# Action endpoints for Streamlit buttons
def create_github_repo(topic: str) -> str:
    return create_poc_repo_from_prompt(topic)

def search_github(query: str) -> list:
    refined_query = rephrase_for_github_search(query)
    return search_github_repositories(refined_query)

def create_jira_ticket(topic: str) -> str:
    return create_core_jira_task_from_prompt(topic)

def schedule_calendar_meeting(topic: str) -> str:
    return schedule_meeting_from_prompt(topic)

if __name__ == "__main__":
    print("🤖 Ask the AI Operating Officer (Satya / Kevin / Pavan)")
    role = input("Choose persona (CEO / CTO / Product): ").strip()
    query = input("Ask your question: ").strip()
    ask_question(query, role)
