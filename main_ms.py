import os
import json
import time
import re
import logging
from urllib.parse import quote_plus
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import requests # Keep for tool files, main.py itself might not use it directly.

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback # For token usage if using OpenAI models

# RAGAS and DeepEval
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# Project-specific modules
from utils.react_validator import validate_trace_output
from agents.ceo_agent import ceo_agent
from agents.cto_agent import cto_agent
from agents.product_agent import product_agent

# Tool imports (ensure these functions exist and are robust)
from tools.github import create_poc_repo_from_prompt # Primary GitHub tool
# from tools.gitlab import create_poc_repo_gitlab_fallback # Fallback if create_poc_repo_from_prompt handles it
from tools.jira import create_core_jira_task_from_prompt # Primary JIRA tool
# from tools.jira_fallback import create_core_jira_task_with_fallback # Fallback if primary handles it
from tools.calender import schedule_meeting # Primary Calendar tool
# from tools.calender_fallback import book_modern_core_clinic_with_fallback # Fallback if primary handles it
from tools.calender_fallback import book_modern_core_clinic_with_fallback
from tools.github import create_poc_repo_from_prompt as create_github_repo

# Setup Logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
LOGS_REASONING_DIR = "logs/reasoning"
VECTORSTORE_DIR = "db"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
RAGAS_RETRY_THRESHOLD = float(os.getenv("RAGAS_RETRY_THRESHOLD", 0.75)) # Score below which to retry retrieval

os.makedirs(LOGS_REASONING_DIR, exist_ok=True)

# --- Initialize Core Components ---
try:
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding_function)
    # Default retriever, can be overridden in ask_question if needed
    #default_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    default_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

except Exception as e:
    logger.critical(f"Failed to initialize core LLM/embedding/vectorstore components: {e}", exc_info=True)
    raise

# --- Helper Functions ---
def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences using common delimiters."""
    if not text:
        return []
    return re.split(r'(?<=[.。!?؟])\s+', text.strip()) # Added common delimiters

# --- Summarization for RAGAS Evaluation ---
def generate_summary_for_evaluation(docs: list, query: str, role: str) -> dict:
    """
    Generates a concise summary from the top retrieved documents,
    strictly based on their content, for RAGAS/DeepEval context-answer evaluation.
    Args:
        docs: List of Langchain Document objects.
        query: The user's query.
        role: The persona role (e.g., "CEO").
    Returns:
        A dictionary containing "eval_summary_text" and "eval_citations".
    """
    logger.info(f"Generating summary for evaluation. Query: '{query[:50]}...', Role: {role}, Docs: {len(docs)}")
    top_docs_for_summary = docs[:3] # Use top 3-5 docs for this summary
    numbered_quotes = []
    source_map = {}
    quote_idx = 1

    for i, doc in enumerate(top_docs_for_summary):
        source_name = doc.metadata.get("source", f"Document {i+1}")
        source_url = doc.metadata.get("url", "#")
        sentences = split_into_sentences(doc.page_content)
        for sent in sentences:
            sent_clean = sent.strip().strip('"')
            if 5 < len(sent_clean.split()) < 150: # Filter reasonable sentence length
                numbered_quotes.append(f'{quote_idx}. "{sent_clean}" — source: {source_name}')
                source_map[quote_idx] = {"name": source_name, "url": source_url}
                quote_idx += 1
            if quote_idx > 7: break # Limit total quotes for the prompt
        if quote_idx > 7: break

    if not numbered_quotes:
        logger.warning("No suitable quotes found in retrieved context for summarization.")
        return {"eval_summary_text": "No relevant information found in the provided context to summarize.", "eval_citations": []}

    quote_text_for_prompt = "\n".join(numbered_quotes)
    summarization_prompt_str = (
        f"You are a factual summarization engine. Based ONLY on the following numbered quotes from {role.title()} perspective, "
        f"write a concise, single-paragraph summary that directly addresses the query: '{query}'.\n"
        f"Strictly adhere to these rules:\n"
        f"1. Use information ONLY from the provided quotes.\n"
        f"2. Cite the quotes used inline with bracketed numbers (e.g., [1], [2]). Example: 'The report shows [1] that AI adoption is growing [2][3].'\n"
        f"3. Do NOT add any external information, opinions, or interpretations.\n"
        f"4. The summary should be dense with information from the quotes.\n\n"
        f"Quotes:\n{quote_text_for_prompt}\n\nFactual Summary:"
    )

    try:
        # Using a separate LLM instance or a specific configuration for summarization if desired
        summarizer_llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.0) # Zero temperature for factual summary
        summary_text = summarizer_llm.invoke(summarization_prompt_str).content.strip()
        logger.info(f"Generated evaluation summary: {summary_text[:100]}...")
    except Exception as e:
        logger.error(f"Error during summary generation for evaluation: {e}", exc_info=True)
        return {"eval_summary_text": "Error generating summary for evaluation.", "eval_citations": []}

    used_quote_ids = sorted(set(map(int, re.findall(r"\[(\d+)\]", summary_text))))
    eval_citations_list = [
        f"[{qid}. {source_map[qid]['name']}]({source_map[qid]['url']})"
        for qid in used_quote_ids if qid in source_map
    ]

    return {"eval_summary_text": summary_text, "eval_citations": eval_citations_list}

# --- Agent Invocation & Orchestration ---
def get_persona_specific_agent_executor(role: str, agent_tools: list[Tool]):
    """Initializes and returns a persona-specific agent executor."""
    logger.debug(f"Initializing agent for role: {role}")
    persona_prompt_template = get_persona_prompt(role) # Fetches ceo_agent(), cto_agent(), etc.

    return initialize_agent(
        tools=agent_tools,
        llm=llm, # Global LLM instance
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=persona_prompt_template,
        verbose=os.getenv("AGENT_VERBOSE", "False").lower() == "true", # Control verbosity via env
        handle_parsing_errors="The previous attempt to use a tool resulted in an error. Please check the observation and try a different approach or rephrase your thought process to conform to the required REACT format.",
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", 5)),
        early_stopping_method="generate",
        return_intermediate_steps=True
    )

# --- Action Suggestion ---
def generate_suggested_actions(query: str, final_agent_answer: str) -> list[str]:
    """Generates suggested next actions based on the agent's final answer."""
    logger.info(f"Generating suggested actions for query: '{query[:50]}...'")
    actions_prompt_str = (
        f"The user asked: '{query}'.\n"
        f"The AI Operating Officer (in persona) responded: '{final_agent_answer}'.\n\n"
        f"Based on this interaction, suggest 2-3 concrete, actionable next steps that a human could take. "
        f"These should be distinct from any tools the AI agent might have already used. "
        f"Format each action clearly starting with '▪ '. Example: ▪ Follow up with the legal team."
    )
    try:
        response = llm.invoke(actions_prompt_str).content.strip()
        suggested_actions_list = [line.strip() for line in response.split('\n') if line.strip().startswith("▪")]
        logger.info(f"Generated {len(suggested_actions_list)} suggested actions.")
        return suggested_actions_list
    except Exception as e:
        logger.error(f"Error generating suggested actions: {e}", exc_info=True)
        return ["▪ Error generating suggested actions."]

def generate_agent_reasoning(chunk_count: int, similarity_score: float, role: str, tools_used: str, tool_action_count: int) -> list[str]:
    """Generates a simple structured summary of agent's reasoning for UI."""
    return [
        f"{role} agent retrieved {chunk_count} chunks from the vectorstore.",
        f"Similarity to top chunk: {similarity_score:.3f}" if similarity_score else "Similarity to top chunk: N/A",
        f"Tools used during REACT trace: {tools_used}.",
        f"{tool_action_count} tool actions were executed in this interaction."
    ]


# --- RAGAS & DeepEval Metric Calculation ---
def calculate_ragas_metrics(query: str, contexts_str_list: list[str], answer_str: str) -> tuple[float, float]:
    """Calculates RAGAS metrics (Faithfulness, Answer Relevancy)."""
    logger.info("Calculating RAGAS metrics...")
    dataset_dict = {'question': [query], 'answer': [answer_str], 'contexts': [contexts_str_list]}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Configure RAGAS LLM if needed (many metrics are LLM-based)
    # from ragas.llms import LangchainLLMWrapper
    # ragas_llm_wrapper = LangchainLLMWrapper(llm) # Use the global llm instance
    # metrics_to_run = [
    #     faithfulness.configure(llm=ragas_llm_wrapper),
    #     answer_relevancy.configure(llm=ragas_llm_wrapper),
    #     context_recall,
    #     context_precision
    # ]
    # For simplicity, using default RAGAS behavior for LLM use in metrics for now
    metrics_to_run = [faithfulness, answer_relevancy]

    try:
        results = ragas_evaluate(dataset, metrics=metrics_to_run)
        faith_score = results['faithfulness'] if 'faithfulness' in results else 0.0
        ans_rel_score = results['answer_relevancy'] if 'answer_relevancy' in results else 0.0

        # ctx_recall = results.get('context_recall', 0.0)
        # ctx_precision = results.get('context_precision', 0.0)

        # Simplified F1-like score; RAGAS offers more nuanced scoring like `ragas_score`
        final_f1_score = (faith_score + ans_rel_score) / 2.0 if faith_score is not None and ans_rel_score is not None else 0.0
        hallucination_rate = 1.0 - faith_score if faith_score is not None else 1.0
        logger.info(f"RAGAS Metrics - F1 (sim): {final_f1_score:.3f}, Hallucination: {hallucination_rate:.3f}")
        return round(final_f1_score, 3), round(hallucination_rate, 3)
    except Exception as e:
        logger.error(f"Error calculating RAGAS metrics: {e}", exc_info=True)
        return 0.0, 1.0 # Default on error

def run_deepeval_assertions(query: str, agent_final_answer: str, retrieved_contexts_str_list: list[str]) -> tuple[bool, dict]:
    """Runs DeepEval assertions and returns pass status and metric results."""
    logger.info("Running DeepEval assertions...")
    test_case = LLMTestCase(
        input=query,
        actual_output=agent_final_answer,
        retrieval_context=retrieved_contexts_str_list
    )
    
    # Define DeepEval metrics with model and threshold
    # Note: DeepEval's LLM-based metrics also need an LLM.
    # It can sometimes pick up Langchain's default or you can pass it.
    # For Ollama with Langchain, ensure DeepEval can use it or wrap it.
    # from deepeval.models import LangchainLLM # If needed for explicit model passing
    # deepeval_llm_wrapper = LangchainLLM(llm) # llm is global ChatOllama
    
    # Using default DeepEval behavior for model selection for now
    metrics_for_deepeval = [
        FaithfulnessMetric(threshold=0.75, include_reason=True), # model=deepeval_llm_wrapper
        AnswerRelevancyMetric(threshold=0.75, include_reason=True), # model=deepeval_llm_wrapper
        ContextualPrecisionMetric(threshold=0.75, include_reason=True) # model=deepeval_llm_wrapper
    ]
    
    passed_status = False
    metric_results_dict = {}
    try:
        assert_test(test_case, metrics=metrics_for_deepeval)
        passed_status = True
        logger.info(f"DeepEval assertions PASSED for query: '{query[:50]}...'")
    except AssertionError as e:
        logger.warning(f"DeepEval assertions FAILED for query: '{query[:50]}...': {e}")
    except Exception as e_other: # Catch other potential errors during DeepEval
        logger.error(f"DeepEval encountered an unexpected error: {e_other}", exc_info=True)

    # Populate metric_results_dict from test_case.metrics_metadata
    if hasattr(test_case, 'metrics_metadata') and test_case.metrics_metadata:
        for m_meta in test_case.metrics_metadata:
            metric_results_dict[m_meta.metric] = {
                "score": float(m_meta.score) if m_meta.score is not None else None,
                "threshold": float(m_meta.threshold) if m_meta.threshold is not None else None,
                "reason": m_meta.reason,
                "success": m_meta.success
            }
    elif not passed_status: # If failed and no metadata, store the assertion error
         metric_results_dict["assertion_error"] = str(e if 'e' in locals() else "Unknown assertion error")


    return passed_status, metric_results_dict


# --- Agent Tool Definitions ---
# These functions wrap the actual tool calls, adding error handling or transformations if needed.
def tool_create_github_repo(prompt_input: str) -> str:
    """Wraps GitHub PoC repo creation tool."""
    logger.info(f"Tool invoked: Create GitHub PoC Repo with input: '{prompt_input}'")
    try:
        # The create_poc_repo_from_prompt should handle its own internal logic including any fallbacks
        # if GitHub fails and it's designed to try GitLab.
        return create_poc_repo_from_prompt(prompt_input)
    except Exception as e:
        logger.error(f"Error in Create GitHub PoC Repo tool: {e}", exc_info=True)
        return f"❌ Error creating repository: {str(e)}"

def tool_create_jira_task(prompt_input: str) -> str:
    """Wraps JIRA task creation tool."""
    logger.info(f"Tool invoked: Create JIRA Task with input: '{prompt_input}'")
    try:
        # create_core_jira_task_from_prompt should handle its own fallbacks if primary JIRA fails
        return create_core_jira_task_from_prompt(prompt_input)
    except Exception as e:
        logger.error(f"Error in Create JIRA Task tool: {e}", exc_info=True)
        return f"❌ Error creating JIRA task: {str(e)}"

def schedule_meeting_from_prompt(prompt: str) -> str:
    return book_modern_core_clinic_with_fallback(prompt)
def tool_schedule_meeting(prompt_input: str) -> str:
    """Wraps meeting scheduling tool."""
    logger.info(f"Tool invoked: Schedule Meeting with input: '{prompt_input}'")
    try:
        # schedule_meeting_from_prompt should handle its own fallbacks
        return schedule_meeting_from_prompt(prompt_input)
    except Exception as e:
        logger.error(f"Error in Schedule Meeting tool: {e}", exc_info=True)
        return f"❌ Error scheduling meeting: {str(e)}"

# Define tools for the agent
agent_tools = [
    Tool(
        name="Create GitHub PoC Repo",
        func=tool_create_github_repo,
        description="Use this to create a GitHub Proof-of-Concept (PoC) repository. Input should be a concise project name or idea based on the user's query or context."
    ),
    Tool(
        name="Create JIRA Task",
        func=tool_create_jira_task,
        description="Use this to create a JIRA task for an engineering team or for follow-up. Input should be a short summary of the task to be created."
    ),
    Tool(
        name="Schedule a Meeting",
        func=tool_schedule_meeting,
        description="Use this to schedule a meeting, for example, with an infrastructure team or for a clinic. Input should be a brief topic or purpose for the meeting."
    ),
]

# --- Main `ask_question` Orchestrator ---
def ask_question(query: str, role: str) -> dict:
    """
    Handles a user query, orchestrates RAG, agent execution, evaluation, and logging.
    """
    request_id = os.urandom(4).hex() # Simple request ID for logging
    logger.info(f"[ReqID: {request_id}] Received query: '{query[:100]}...', Role: {role}")
    start_time = time.time()
    
    # Initialize retriever for this specific call (allows dynamic changes if needed)
    current_retriever = default_retriever 
    
    try:
        # 1. Document Retrieval
        docs = current_retriever.invoke(query, filter={"persona": role.lower()})
        if not docs:
            logger.warning(f"[ReqID: {request_id}] No docs found for persona '{role}'. Trying generic retrieval.")
            docs = current_retriever.invoke(query)
        
        context_chunks_str_list = [doc.page_content for doc in docs]

        if not context_chunks_str_list:
            logger.error(f"[ReqID: {request_id}] No relevant documents found for query.")
            # ... (return structure for no docs - kept from previous good version)
            return {
                "answer": "I'm sorry, I couldn't find relevant information to answer your question at this time.",
                "paraphrased_blocks": [], "actions": [], "reasoning": {"explanation": "No documents retrieved from vector store."},
                "score": 0.0, "latency": (time.time() - start_time) * 1000, "hallucination": 1.0,
                "deepeval_passed": False, "citations": [], "trace": {"error": "No documents retrieved"}, "context_chunks": [],
                "query": query, "role": role, "agent_raw_output": "No context available."
            }
        logger.info(f"[ReqID: {request_id}] Retrieved {len(docs)} documents.")

        # 2. Generate Summary for RAGAS Evaluation (uses top N docs)
        eval_summary_obj = generate_summary_for_evaluation(docs, query, role)
        eval_summary_text = eval_summary_obj["eval_summary_text"]
        eval_citations_list = eval_summary_obj["eval_citations"]

        # 3. Calculate RAGAS Metrics (on the evaluation summary)
        ragas_f1_score, hallucination_metric = calculate_ragas_metrics(query, context_chunks_str_list, eval_summary_text)

        # 4. Conditional Retrieval Retry (if RAGAS score is low)
        if ragas_f1_score < RAGAS_RETRY_THRESHOLD:
            logger.info(f"[ReqID: {request_id}] Initial RAGAS F1 {ragas_f1_score} < {RAGAS_RETRY_THRESHOLD}. Attempting expanded retrieval.")
            #fallback_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10, "fetch_k": 30})
            fallback_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

            fallback_docs = fallback_retriever.invoke(query)
            fallback_chunks_str_list = [doc.page_content for doc in fallback_docs]

            if fallback_chunks_str_list:
                logger.info(f"[ReqID: {request_id}] Retrieved {len(fallback_docs)} fallback documents.")
                temp_eval_summary_obj = generate_summary_for_evaluation(fallback_docs, query, role)
                temp_eval_summary_text = temp_eval_summary_obj["eval_summary_text"]
                
                temp_ragas_f1, temp_hallucination = calculate_ragas_metrics(query, fallback_chunks_str_list, temp_eval_summary_text)
                logger.info(f"[ReqID: {request_id}] Fallback retrieval RAGAS F1: {temp_ragas_f1}")

                if temp_ragas_f1 > ragas_f1_score:
                    logger.info(f"[ReqID: {request_id}] RAGAS F1 improved from {ragas_f1_score} to {temp_ragas_f1} with fallback.")
                    docs = fallback_docs # Update docs
                    context_chunks_str_list = fallback_chunks_str_list # Update context for agent and DeepEval
                    eval_summary_text = temp_eval_summary_text # Update summary used for RAGAS score reporting
                    eval_citations_list = temp_eval_summary_obj["eval_citations"]
                    ragas_f1_score = temp_ragas_f1
                    hallucination_metric = temp_hallucination
        
        # 5. Initialize and Invoke Persona-Specific Agent
        persona_agent_executor = get_persona_specific_agent_executor(role, agent_tools)
        
        # Prepare limited context for the agent's prompt
        formatted_context_for_agent = "\n\n---\n\n".join(
            [f"Retrieved Context Snippet {i+1} (Source: {docs[i].metadata.get('source','N/A')}):\n{docs[i].page_content}"
             for i in range(min(len(docs), 3))] # Agent sees top 3 docs for its reasoning
        )
        logger.debug(f"[ReqID: {request_id}] Context for agent: {formatted_context_for_agent[:200]}...")

        combined_prompt_input = f"Question: {query}\n\nContext:\n{formatted_context_for_agent}"
        raw_agent_trace = persona_agent_executor.invoke({"input": combined_prompt_input})

        agent_final_answer_str = raw_agent_trace.get("output", "The agent could not determine a final answer.")
        logger.info(f"[ReqID: {request_id}] Agent final answer: {agent_final_answer_str[:200]}...")

        # 6. Process Agent Output for Display
        # The agent is prompted to use 🔹 and 🧠. The UI should render these.
        # For `paraphrased_blocks`, we assume the agent's output is the primary source.
        paraphrased_blocks_for_display = [{"type": "agent_output", "text": agent_final_answer_str}]
        # Citations from agent are harder to parse reliably unless it uses a strict format.
        # For now, UI will display eval_citations_list if needed, or agent's 🔹 markers.

        # 7. Generate Suggested Actions (based on agent's final answer)
        suggested_actions_list = generate_suggested_actions(query, agent_final_answer_str)

        # 8. Collate Executed Tool Actions from Agent Trace
        agent_intermediate_steps = raw_agent_trace.get("intermediate_steps", [])
        executed_tool_actions_list = []
        tools_used_by_agent_set = set()
        for step in agent_intermediate_steps:
            if isinstance(step, tuple) and hasattr(step[0], "tool"):
                tool_name = step[0].tool
                observation = step[1]
                tools_used_by_agent_set.add(tool_name)
                status_emoji = "✅ Success" if not (isinstance(observation, str) and ("fail" in observation.lower() or "❌" in observation.lower() or "error" in observation.lower())) else "❌ Failed"
                executed_tool_actions_list.append(f"▪ Agent Action: {status_emoji} — {tool_name} → {str(observation)[:150]}")
        
        tools_used_str = ", ".join(tools_used_by_agent_set) if tools_used_by_agent_set else "None"
        combined_actions_for_ui = executed_tool_actions_list + suggested_actions_list

        # 9. Run DeepEval Assertions (on agent's final answer)
        deepeval_passed_bool, deepeval_metrics_map = run_deepeval_assertions(query, agent_final_answer_str, context_chunks_str_list)

        # 10. Construct Reasoning Details for Logging & UI
        similarity_score_val = 0.0 # Placeholder, implement if needed for display
        if docs and context_chunks_str_list:
             query_emb = embedding_function.embed_query(query)
             first_doc_emb = embedding_function.embed_documents([context_chunks_str_list[0]])[0]
             query_emb_np = np.array(query_emb)
             first_doc_emb_np = np.array(first_doc_emb)
             if query_emb_np.shape == first_doc_emb_np.shape and np.linalg.norm(query_emb_np) != 0 and np.linalg.norm(first_doc_emb_np) != 0 :
                similarity_score_val = np.dot(query_emb_np, first_doc_emb_np) / (np.linalg.norm(query_emb_np) * np.linalg.norm(first_doc_emb_np))


        reasoning_data_dict = {
            "retrieved_chunks_count": len(context_chunks_str_list),
            "similarity_to_top_chunk": f"{similarity_score_val:.3f}" if similarity_score_val else "N/A",
            "tools_used_by_agent": tools_used_str,
            "citations_from_eval_summary": eval_citations_list, # Citations from the summary used for RAGAS
            "ragas_f1_evaluated_on_summary": ragas_f1_score,
            "hallucination_ragas_evaluated_on_summary": hallucination_metric,
            "deepeval_passed_on_agent_answer": deepeval_passed_bool,
            "deepeval_metrics_results": deepeval_metrics_map,
            "constructed_explanation_for_ui": generate_agent_reasoning(
                len(context_chunks_str_list), similarity_score_val,
                role.title(), tools_used_str, len(executed_tool_actions_list)
            )
        }
        
        latency_ms_val = (time.time() - start_time) * 1000
        logger.info(f"[ReqID: {request_id}] Query processed. Latency: {latency_ms_val:.0f}ms. RAGAS F1: {ragas_f1_score}. DeepEval Passed: {deepeval_passed_bool}")

        # 11. Final Return Structure
        final_result = {
            "answer": agent_final_answer_str,
            "paraphrased_blocks": paraphrased_blocks_for_display,
            "actions": combined_actions_for_ui,
            "reasoning": reasoning_data_dict,
            "score": ragas_f1_score, # RAGAS F1 (on eval_summary_text)
            "latency": latency_ms_val,
            "hallucination": hallucination_metric, # RAGAS Hallucination (on eval_summary_text)
            "deepeval_passed": deepeval_passed_bool,
            "citations": eval_citations_list, # Using eval citations for display, agent should embed 🔹
            "trace": raw_agent_trace,
            "context_chunks": context_chunks_str_list,
            "query": query,
            "role": role,
            "eval_summary_for_ragas": eval_summary_text # For transparency / debugging
        }
        
        # 12. Save Reasoning Log
        save_reasoning_json(
            query, role, context_chunks_str_list, raw_agent_trace, agent_final_answer_str,
            combined_actions_for_ui, reasoning_data_dict, ragas_f1_score, latency_ms_val,
            hallucination_metric, deepeval_passed_bool, "N/A_feedback_in_main" # Feedback usually from UI
        )
        return final_result

    except Exception as e_ask:
        logger.critical(f"[ReqID: {request_id}] Critical error in ask_question for query '{query[:50]}...': {e_ask}", exc_info=True)
        # ... (consistent error structure - kept from previous good version)
        return {
            "answer": f"An unexpected critical error occurred: {str(e_ask)}",
            "paraphrased_blocks": [], "actions": [], "reasoning": {"explanation": f"Critical Error: {str(e_ask)}", "traceback": traceback.format_exc() if 'traceback' in locals() else "No traceback"},
            "score": 0.0, "latency": (time.time() - start_time) * 1000 if 'start_time' in locals() else -1, "hallucination": 1.0,
            "deepeval_passed": False, "citations": [], "trace": {"error": str(e_ask)}, "context_chunks": [],
            "query": query, "role": role, "agent_raw_output": "Critical error before/during agent execution."
        }

# --- Persona Prompt Getters ---
def get_persona_prompt(role: str) -> PromptTemplate:
    """Loads and returns the PromptTemplate for the specified role."""
    logger.debug(f"Loading prompt for role: {role}")
    if role.lower() == "ceo":
        return ceo_agent()  # Assumes ceo_agent() returns a PromptTemplate instance
    elif role.lower() == "cto":
        return cto_agent()
    elif role.lower() in ["product", "evp"]:
        return product_agent()
    else:
        logger.error(f"Unknown role specified: {role}. Defaulting to CEO.")
        # raise ValueError(f"Unknown role: {role}") # Or default
        return ceo_agent()


# --- JSON Logging ---
def save_reasoning_json(
    query: str, role: str, context_chunks_list: list[str], raw_agent_trace_dict: dict,
    final_agent_answer_str: str, combined_actions_list: list[str],
    reasoning_details_dict: dict, ragas_f1_score_val: float, latency_ms_val: float,
    hallucination_metric_val: float, deepeval_pass_status_bool: bool, user_feedback_str: str
) -> None:
    """Saves the detailed reasoning and evaluation log for a query."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for uniqueness
    
    # Validate the agent's final answer for REACT structure markers (🔹, 🧠)
    # This assumes validate_trace_output is designed for the agent's final textual answer
    format_validation_results = validate_trace_output(final_agent_answer_str)

    log_payload = {
        "timestamp": timestamp,
        "persona_role": role,
        "user_query": query,
        "final_agent_answer": final_agent_answer_str,
        "retrieved_contexts": context_chunks_list,
        "tools_used_by_agent": reasoning_details_dict.get("tools_used_by_agent", "None"),
        "actions_for_ui": combined_actions_list,
        "evaluation_metrics": {
            "ragas_f1_on_eval_summary": ragas_f1_score_val,
            "ragas_hallucination_on_eval_summary": hallucination_metric_val,
            "deepeval_passed_on_agent_answer": deepeval_pass_status_bool,
            "deepeval_detailed_results": reasoning_details_dict.get("deepeval_metrics_results", {}),
            "latency_ms": latency_ms_val,
        },
        "user_feedback": user_feedback_str,
        "agent_reasoning_summary_for_ui": reasoning_details_dict.get("constructed_explanation_for_ui", []),
        "full_reasoning_details": reasoning_details_dict, # Contains more detailed eval summary citations etc.
        "final_answer_format_validation": {
            "issues": format_validation_results.get("issues", []),
            "lifted_present": format_validation_results.get("lifted_present", False),
            "generated_present": format_validation_results.get("generated_present", False),
            "react_steps_in_answer": format_validation_results.get("react_steps", 0)
        },
        "raw_langchain_agent_trace": raw_agent_trace_dict # The full trace from agent.invoke
    }

    file_path = os.path.join(LOGS_REASONING_DIR, f"reasoning_log_{timestamp}.json")
    try:
        with open(file_path, "w", encoding='utf-8') as f: # Specify encoding
            json.dump(log_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Reasoning log saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save reasoning log to {file_path}: {e}", exc_info=True)


# --- GitHub Search Specific ---
def rephrase_for_github_search(query: str) -> str:
    """Rephrases a natural language query into a concise GitHub search query."""
    # ... (implementation from your last version is fine, ensure llm is accessible)
    logger.debug(f"Rephrasing for GitHub search: '{query[:50]}...'")
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
    try:
        response = llm.invoke(prompt).content.strip().replace("`", "")
        return response.splitlines()[0] if response.splitlines() else query # Fallback to original query
    except Exception as e:
        logger.error(f"Error rephrasing for GitHub search: {e}", exc_info=True)
        return query # Fallback to original query on error


def search_github_repositories(refined_query: str) -> list[str]:
    """Searches GitHub repositories for the refined query."""
    # ... (implementation from your last version is generally fine, ensure robust error handling)
    logger.info(f"Searching GitHub repositories with query: '{refined_query}'")
    # Ensure requests and quote_plus are imported (typically at top of file)
    # from urllib.parse import quote_plus
    # import requests

    headers = {"Accept": "application/vnd.github+json", "User-Agent": "AI-Operating-Officer"}
    encoded_query = quote_plus(refined_query)
    params = {"q": encoded_query, "sort": "stars", "order": "desc", "per_page": 3}

    try:
        response = requests.get("https://api.github.com/search/repositories", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items: return ["ℹ️ No matching GitHub repositories found."]
        return [f"[{repo['full_name']}]({repo['html_url']}) - ⭐{repo.get('stargazers_count',0)}" for repo in items]
    except requests.exceptions.HTTPError as e_http:
        err_msg = f"❌ GitHub search failed (HTTP {e_http.response.status_code}): {str(e_http.response.text)[:100]}"
        logger.warning(err_msg)
        return [err_msg]
    except requests.exceptions.RequestException as e_req: # Catches DNS, Connection, Timeout etc.
        err_msg = f"❌ GitHub search failed (Network/Request Error): {str(e_req)[:100]}"
        logger.warning(err_msg)
        return [err_msg]
    except Exception as e_gen:
        err_msg = f"❌ GitHub search failed (Unexpected Error): {str(e_gen)[:100]}"
        logger.error(err_msg, exc_info=True)
        return [err_msg]

# --- CLI Execution ---
if __name__ == "__main__":
    logger.info("AI Operating Officer CLI started.")
    print("🤖 Ask the AI Operating Officer (CEO / CTO / Product)")
    try:
        selected_role = input("Choose persona (CEO / CTO / Product): ").strip().upper()
        if selected_role not in ["CEO", "CTO", "PRODUCT"]:
            logger.warning(f"Invalid persona '{selected_role}'. Defaulting to CEO.")
            selected_role = "CEO"
        
        user_query = input(f"Ask {selected_role}: ").strip()
        if not user_query:
            print("No query entered. Exiting.")
            logger.info("No query entered by user. Exiting CLI.")
            exit()

        # Example of using OpenAI token counter if you were using OpenAI models
        # with get_openai_callback() as cb:
        # result = ask_question(user_query, selected_role)
        #     print(f"Total Tokens: {cb.total_tokens}")
        #     print(f"Prompt Tokens: {cb.prompt_tokens}")
        #     print(f"Completion Tokens: {cb.completion_tokens}")
        #     print(f"Total Cost (USD): ${cb.total_cost}")
        
        result = ask_question(user_query, selected_role)
        
        print("\n───────────────── AI Operating Officer Response ─────────────────")
        print(f"🗣️ Persona: {result.get('role', 'N/A')}")
        print(f"❓ Query: {result.get('query', 'N/A')}")
        print("\n📝 Agent's Final Answer:")
        print(result.get("answer", "No answer provided."))
        
        # Citations displayed are from the eval_summary, UI should parse agent's 🔹 markers
        if result.get("citations"):
            print("\n📚 Footnotes (from evaluation summary for RAGAS):")
            for c_item in result["citations"]:
                print(f"- {c_item}")
        
        print("\n🔧 Suggested & Executed Actions:")
        if result.get("actions"):
            for action_item_str in result["actions"]:
                print(action_item_str)
        else:
            print("▪ (No actions were executed by agent or suggested.)")

        reasoning_output = result.get("reasoning", {})
        print("\n🧠 Agent’s Reasoning Insights (Constructed Summary):")
        if isinstance(reasoning_output.get("constructed_explanation_for_ui"), list):
            for r_line_str in reasoning_output["constructed_explanation_for_ui"]:
                print(r_line_str)
        else:
            print(reasoning_output.get("constructed_explanation_for_ui", "No detailed reasoning summary available."))
        
        print("\n📊 Evaluation Metrics Overview:")
        print(f"- RAGAS F1 Score (on eval summary): {result.get('score', 0.0):.3f}")
        print(f"- RAGAS Hallucination Rate (on eval summary): {result.get('hallucination', 1.0):.3f}")
        print(f"- DeepEval Test Passed (on agent answer): {result.get('deepeval_passed', False)}")
        
        deepeval_details = reasoning_output.get("deepeval_metrics_results", {})
        if isinstance(deepeval_details, dict) and deepeval_details:
            print("  DeepEval Metric Details:")
            for metric_name_str, de_res_dict in deepeval_details.items():
                if isinstance(de_res_dict, dict):
                    score_str = f"{de_res_dict.get('score', 'N/A'):.3f}" if isinstance(de_res_dict.get('score'), float) else de_res_dict.get('score', 'N/A')
                    reason_str = de_res_dict.get('reason', 'N/A') if de_res_dict.get('reason') else 'N/A'
                    success_str = de_res_dict.get('success', 'N/A')
                    print(f"    - {metric_name_str.capitalize()}: Score {score_str}, Success: {success_str}, Reason: {reason_str}")
                else:
                     print(f"    - {metric_name_str.capitalize()}: {str(de_res_dict)}")


        print(f"- Latency: {result.get('latency', -1):.0f}ms")
        print(f"\nℹ️ Full reasoning log saved (see logs/reasoning directory).")

    except Exception as e_cli:
        logger.critical(f"Unhandled error in CLI main block: {e_cli}", exc_info=True)
        print(f"\nAn unexpected error occurred in the application: {e_cli}")

search_github = search_github_repositories
create_github_repo = tool_create_github_repo
create_jira_ticket = tool_create_jira_task
schedule_calendar_meeting = tool_schedule_meeting
