import streamlit as st
import os
import logging
import json
import datetime
from typing import Dict, Any, List, Tuple, Optional

# Assuming main.py is in the same directory or configured in PYTHONPATH
# from main_ms import (
#     ask_question,
#     create_github_repo, # Assuming these are still direct endpoints for buttons
#     search_github,
#     create_jira_ticket,
#     schedule_calendar_meeting,
#     save_reasoning_json # If feedback saving is still managed here
# )
# from main_ms import ask_question, search_github, create_jira_ticket, schedule_calendar_meeting, save_reasoning_json
from main_ms import (
    ask_question,
    search_github,
    create_github_repo,
    create_jira_ticket,
    schedule_calendar_meeting,
    save_reasoning_json
)

from tools.github import create_poc_repo_from_prompt as create_github_repo

# Configure logger for the Streamlit app
logger = logging.getLogger(__name__)
# BasicConfig should ideally be in a central place if not already set by main.py when imported
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(page_title="AI Operating Officer", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions for UI ---
def display_chat_message(author_role: str, text: str, avatar: Optional[str] = None):
    """Displays a chat message with an avatar."""
    if avatar:
        st.markdown(f"<div style='display:flex; align-items:center; margin-bottom:10px;'><img src='{avatar}' width=30 height=30 style='border-radius:50%; margin-right:10px;'><div><b>{author_role}</b><br>{text}</div></div>", unsafe_allow_html=True)
    else:
        with st.chat_message(author_role):
            st.markdown(text)

def render_paraphrased_blocks(paraphrased_blocks: List[Dict[str, str]]):
    """Renders paraphrased blocks, looking for 🔹 and 🧠 markers if agent output is structured that way."""
    if not paraphrased_blocks:
        st.markdown("_No specific paraphrased blocks provided._")
        return

    # The new main.py's `paraphrased_blocks` from `generate_paraphrased_answer_from_agent_output`
    # currently puts the whole agent_final_answer as one block.
    # The agent itself is prompted to use 🔹 and 🧠.
    # This function should ideally parse those from the text or the UI should handle markdown.
    for block in paraphrased_blocks:
        text_content = block.get("text", "")
        # A more sophisticated renderer would parse text_content for 🔹 and 🧠 and style them.
        # For now, just display the text. The agent's answer should have the markers.
        st.markdown(text_content) # Let Streamlit's markdown handle potential markers

def render_actions(actions_list: List[str]):
    """Renders suggested and executed actions."""
    if actions_list:
        for action_item in actions_list:
            st.markdown(action_item) # Actions are pre-formatted with ▪ and status
    else:
        st.markdown("▪ _(No actions were executed by the agent or suggested.)_")

def render_citations(citations_list: List[str]):
    """Renders citations if available."""
    if citations_list:
        st.markdown("**📚 Footnotes/Citations (from evaluation summary):**")
        for citation_item in citations_list:
            st.markdown(f"- {citation_item}")

def render_reasoning(reasoning_data: Dict[str, Any]):
    if not reasoning_data:
        st.markdown("_No reasoning data available._")
        return

    constructed_explanation = reasoning_data.get("constructed_explanation_for_ui", [])
    if isinstance(constructed_explanation, list):
        for line in constructed_explanation:
            st.markdown(f"• {line}")
    elif isinstance(constructed_explanation, str):
        st.markdown(f"• {constructed_explanation}")

    st.markdown("---")
    f1_val = reasoning_data.get('ragas_f1_evaluated_on_summary', 'N/A')
    st.markdown(f"**RAGAS F1 (on eval summary):** `{float(f1_val):.3f}`" if isinstance(f1_val, (int, float)) else f"**RAGAS F1 (on eval summary):** `{f1_val}`")

    halluc_val = reasoning_data.get('hallucination_ragas_evaluated_on_summary', 'N/A')
    st.markdown(f"**RAGAS Hallucination (on eval summary):** `{float(halluc_val):.3f}`" if isinstance(halluc_val, (int, float)) else f"**RAGAS Hallucination (on eval summary):** `{halluc_val}`")

    st.markdown(f"**DeepEval Passed (on agent answer):** `{reasoning_data.get('deepeval_passed_on_agent_answer', 'N/A')}`")

    deepeval_details = reasoning_data.get("deepeval_metrics_results", {})
    if isinstance(deepeval_details, dict) and deepeval_details:
        st.markdown("**DeepEval Metric Details:**")
        for metric_name, de_res in deepeval_details.items():
            if isinstance(de_res, dict):
                score_str = f"{de_res.get('score', 'N/A'):.3f}" if isinstance(de_res.get('score'), float) else de_res.get('score', 'N/A')
                reason_str = de_res.get('reason', 'N/A') if de_res.get('reason') else 'N/A'
                success_str = de_res.get('success', 'N/A')
                st.markdown(f"  - **{metric_name.replace('_', ' ').title()}**: Score `{score_str}`, Success: `{success_str}`\n    Reason: _{reason_str}_")
            elif metric_name == "assertion_error":
                st.markdown(f"  - **Assertion Error**: _{str(de_res)}_")

    st.markdown(f"**Retrieved Chunks:** `{reasoning_data.get('retrieved_chunks_count', 'N/A')}`")
    st.markdown(f"**Similarity (Top Chunk):** `{reasoning_data.get('similarity_to_top_chunk', 'N/A')}`")
    st.markdown(f"**Tools Used by Agent:** `{reasoning_data.get('tools_used_by_agent', 'None')}`")


# --- Main Application ---
def run_app():
    """Main function to run the Streamlit application."""
    st.title("🤖 AI Operating Officer")
    st.caption("Powered by Langchain, RAGAS, DeepEval, and custom agents.")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # List of tuples: (role, query, result_dict)
    if "current_result" not in st.session_state: # Stores the full result dict
        st.session_state.current_result = None
    if "feedback_key_suffix" not in st.session_state: # To ensure feedback buttons are unique per response
        st.session_state.feedback_key_suffix = 0

    # --- Sidebar for Persona and Input ---
    with st.sidebar:
        st.header("Query Input")
        persona_options = ["CEO", "CTO", "Product"]
        selected_persona = st.selectbox("Choose Persona:", persona_options, key="persona_select")
        
        user_query = st.text_area("Ask your question:", height=150, key="query_input", placeholder="e.g., How can we improve developer productivity using AI?")
        
        submit_button = st.button("🚀 Get Insights", type="primary", use_container_width=True)

    # --- Main Interaction Area ---
    if submit_button and user_query and selected_persona:
        st.session_state.feedback_key_suffix += 1 # New response, new feedback widgets
        with st.spinner(f"🤖 {selected_persona} is thinking... (this may take a moment)"):
            try:
                logger.info(f"UI: Submitting query for persona {selected_persona}: '{user_query[:50]}...'")
                # Call the main orchestration function
                result_data = ask_question(user_query, selected_persona.upper()) # Ensure role is uppercase as expected by main
                st.session_state.current_result = result_data
                # Add to chat history (simplified, could store more detail)
                st.session_state.chat_history.append(
                    (selected_persona, user_query, result_data.get("answer", "No answer generated."))
                )

            except Exception as e:
                logger.error(f"Error during ask_question call from UI: {e}", exc_info=True)
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.current_result = None # Clear previous result on error

    # --- Display Current Result ---
    if st.session_state.current_result:
        result = st.session_state.current_result
        feedback_key_base = f"feedback_{st.session_state.feedback_key_suffix}"

        st.subheader(f"🎙️ {result.get('role', 'Selected Persona')}'s Perspective:")
        render_paraphrased_blocks(result.get("paraphrased_blocks", []))
        render_citations(result.get("citations", []))
        
        st.subheader("💡 Suggested & Executed Actions:")
        render_actions(result.get("actions", []))

        with st.expander("🔍 Show Agent's Reasoning & Evaluation Metrics", expanded=False):
            render_reasoning(result.get("reasoning", {}))
            st.markdown(f"**Overall Latency:** `{result.get('latency', 'N/A'):.0f}ms`")

        # --- Automation Buttons (if actions are suitable for direct trigger) ---
        # This section might need refinement based on what `result.get("actions")` contains.
        # For now, keeping the original button structure that calls separate functions.
        # Ideally, the agent's identified actions should be presented more dynamically.
        st.subheader("⚙️ Trigger Automation (Manual)")
        col1, col2, col3, col4 = st.columns(4)
        
        current_query_for_tools = result.get("query", "No active query") # Use the query that generated the result

        with col1:
            if st.button("📁 Create GitHub Repo", key=f"{feedback_key_base}_gh", use_container_width=True):
                with st.spinner("Creating GitHub repo..."):
                    try:
                        repo_link = create_github_repo(current_query_for_tools) # Uses the main query
                        st.success(f"GitHub Repo Action: {repo_link}")
                    except Exception as e_tool:
                        st.error(f"GitHub Repo creation failed: {e_tool}")
        with col2:
            if st.button("🔍 Search GitHub", key=f"{feedback_key_base}_gh_search", use_container_width=True):
                with st.spinner("Searching GitHub..."):
                    try:
                        search_results = search_github(current_query_for_tools)
                        st.markdown("**GitHub Search Results:**")
                        for res_item in search_results: st.markdown(f"- {res_item}")
                    except Exception as e_tool:
                        st.error(f"GitHub Search failed: {e_tool}")
        with col3:
            if st.button("🗂️ Create JIRA Task", key=f"{feedback_key_base}_jira", use_container_width=True):
                with st.spinner("Creating JIRA Task..."):
                    try:
                        jira_result = create_jira_ticket(current_query_for_tools)
                        st.success(f"JIRA Task Action: {jira_result}")
                    except Exception as e_tool:
                        st.error(f"JIRA Task creation failed: {e_tool}")
        with col4:
            if st.button("📅 Schedule Meeting", key=f"{feedback_key_base}_cal", use_container_width=True):
                with st.spinner("Scheduling meeting (link)..."):
                    try:
                        meeting_info = schedule_calendar_meeting(current_query_for_tools)
                        st.success(f"Calendar Action: {meeting_info}") # Assumes it returns a link or info
                    except Exception as e_tool:
                        st.error(f"Calendar scheduling failed: {e_tool}")
        
        # --- Feedback Section ---
        st.subheader("🗣️ Was this response helpful?")
        fb_col1, fb_col2, fb_col3 = st.columns(3)

        if fb_col1.button("✅ Yes", key=f"{feedback_key_base}_yes", use_container_width=True):
            save_reasoning_json_feedback("✅")
            st.toast("Thank you for your feedback!", icon="👍")

        elif fb_col2.button("🤔 Needs Improvement", key=f"{feedback_key_base}_improve", use_container_width=True):
            save_reasoning_json_feedback("🤔")
            st.toast("Thanks! We'll use this to improve.", icon="💡")
            with st.spinner("🔄 Re-evaluating your question..."):
                try:
                    st.session_state.feedback_key_suffix += 1
                    previous_query = st.session_state.current_result.get("query", "")
                    previous_role = st.session_state.current_result.get("role", "")
                    result_data = ask_question(previous_query, previous_role)
                    st.session_state.current_result = result_data
                except Exception as e:
                    st.error(f"Retry failed: {e}")

        elif fb_col3.button("❌ No", key=f"{feedback_key_base}_no", use_container_width=True):
            save_reasoning_json_feedback("❌")
            st.toast("We appreciate the input.", icon="📝")
            with st.spinner("🔄 Re-evaluating your question..."):
                try:
                    st.session_state.feedback_key_suffix += 1
                    previous_query = st.session_state.current_result.get("query", "")
                    previous_role = st.session_state.current_result.get("role", "")
                    result_data = ask_question(previous_query, previous_role)
                    st.session_state.current_result = result_data
                except Exception as e:
                    st.error(f"Retry failed: {e}")


    # --- Chat History (Optional Display) ---
    if st.session_state.chat_history:
        with st.expander("📜 Recent Interactions", expanded=False):
            for i, (role, q, ans_text) in enumerate(reversed(st.session_state.chat_history[-5:])): # Show last 5
                st.markdown(f"**{i+1}. {role} asked:** _{q}_")
                st.markdown(f"**Response:** {ans_text[:150]}...") # Show snippet
                st.markdown("---")

def save_reasoning_json_feedback(feedback_value: str):
    """Saves feedback to the last generated reasoning log if available."""
    # This function needs access to all the parameters `save_reasoning_json` expects.
    # It's tricky to call it from here without re-fetching/storing all that data.
    # A better approach would be to:
    # 1. Generate a unique ID for each `ask_question` result.
    # 2. When feedback is given, send this ID and feedback_value to an API endpoint.
    # 3. The API endpoint then loads the corresponding JSON log and appends/updates feedback.
    # For now, this is a placeholder, as directly calling save_reasoning_json is complex here.
    logger.info(f"Feedback received: {feedback_value}. (Note: Full log update not implemented directly in UI feedback button).")
    # To actually save it, you'd need to retrieve all params for save_reasoning_json
    # from st.session_state.current_result and call it.
    current_res = st.session_state.current_result
    if current_res:
        try:
            # This is a simplified call, assuming `main.save_reasoning_json` can handle partial updates
            # or that you reconstruct all necessary arguments.
            # This specific call will likely fail if save_reasoning_json isn't designed for it.
            # The main `save_reasoning_json` is already called in `ask_question`.
            # This would be about *updating* that log or creating a separate feedback event.
            
            # For simplicity in this example, we assume the log was already saved by `ask_question`.
            # This function would ideally update that specific log file.
            # Let's simulate finding the latest log and appending feedback.
            log_dir = "logs/reasoning"
            log_files = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("reasoning_log_") and f.endswith(".json")])
            if log_files:
                latest_log_file = log_files[-1]
                # Crude update: just append feedback to a field or as a new key.
                # This is NOT robust for concurrent use or complex updates.
                try:
                    with open(latest_log_file, 'r+') as f:
                        log_data = json.load(f)
                        log_data['user_feedback_ui'] = feedback_value 
                        log_data['feedback_timestamp_ui'] = datetime.now().isoformat()
                        f.seek(0)
                        json.dump(log_data, f, indent=2, ensure_ascii=False)
                        f.truncate()
                    logger.info(f"Appended UI feedback to {latest_log_file}")
                except Exception as e_log_update:
                     logger.error(f"Could not update log file {latest_log_file} with UI feedback: {e_log_update}")
            else:
                logger.warning("No reasoning log found to append UI feedback to.")

        except Exception as e:
            logger.error(f"Error trying to save feedback via UI: {e}")
    else:
        logger.warning("No current result in session state to associate feedback with.")


if __name__ == "__main__":
    run_app()
