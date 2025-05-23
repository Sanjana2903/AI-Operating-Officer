import streamlit as st
import os
from main import ask_question, create_github_repo, create_jira_ticket, schedule_calendar_meeting, save_reasoning_json, search_github

st.set_page_config(page_title="AI Operating Officer", layout="wide")
st.title("🤖 AI Operating Officer")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_role" not in st.session_state:
    st.session_state.last_role = ""

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

query = st.text_area("Ask your question:", height=150)
role = st.selectbox("Choose Persona", ["CEO", "CTO", "Product"])
submit = st.button("Submit")

if submit and query:
    with st.spinner("Running AI Agent..."):
        result = ask_question(query, role.lower())

    st.session_state.last_result = result
    st.session_state.last_query = query
    st.session_state.last_role = role
    st.session_state.chat_history.append((role, query, result))

if st.session_state.last_result:
    result = st.session_state.last_result

    st.subheader("🧠 Paraphrased Answer")
    for block in result.get("paraphrased_blocks", []):
        if block["type"] == "lifted":
            quote_text = block["text"].strip()
            footnote = block.get("footnote", "")
            st.markdown(f"> {quote_text} <sup>[{footnote}]</sup>")
        elif block["type"] == "generated":
            st.markdown(block["text"])

    if result["citations"]:
        st.markdown("**📚 Footnotes:**")
        for c in result["citations"]:
            st.markdown(f"- {c}")

    st.subheader("🔧 Suggested Actions")
    if result["actions"]:
        for action in result["actions"]:
            st.markdown(action)
    else:
        st.markdown("▪ (No actions triggered)")

    st.subheader("🧠 Agent’s Reasoning")
    if isinstance(result["reasoning"]["explanation"], list):
        for r in result["reasoning"]["explanation"]:
            st.markdown(f"• {r}")
    else:
        st.warning(result["reasoning"]["explanation"])


    st.subheader("📊 Trace Metrics")
    col1, col2, col3, col4 = st.columns(4)

    # Format hallucination % (0.18 → 18%)
    hallucination_percent = round(result['hallucination'] * 100, 1)

    # Show latency, RAGAS F1, hallucination rate, and DeepEval pass/fail
    col1.metric("Latency", f"{result['latency']:.0f} ms")
    col2.metric("RAGAS F1 Score", f"{result['score']:.2f}")
    col3.metric("Hallucination", f"{hallucination_percent} %")
    col4.metric("DeepEval", "Pass" if result.get("deepeval_passed", False) else "Fail")


    st.subheader("⚙️ Trigger Automation")
    col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     if st.button("📁 Create GitHub Repo"):
    #         repo_link = create_github_repo(st.session_state.last_query)
    #         st.success(f"GitHub Repo: {repo_link}")
    #         result["actions"].append(f"▪ ✅ Success — GitHub → {repo_link}")
    with col1:
        if st.button("📁 Create GitHub Repo"):
            with st.spinner("Creating GitHub repo..."):
                try:
                    repo_link = create_github_repo(st.session_state.last_query)
                    if isinstance(repo_link, str) and "http" in repo_link:
                        st.success(f"GitHub Repo: {repo_link}")
                        result["actions"].append(f"▪ ✅ Success — GitHub → {repo_link}")
                    else:
                        raise ValueError(repo_link)
                except Exception as e:
                    github_fallback = os.getenv("GITHUB_MANUAL_URL", "https://github.com/new")
                    st.error(f"❌ GitHub repo creation failed: {str(e)}")
                    st.markdown(f"[Manually create repo →]({github_fallback})", unsafe_allow_html=True)
    with col2:                
        if st.button("🔍 Search GitHub Repos"):
            with st.spinner("Searching GitHub..."):
                repos = search_github(st.session_state.last_query)
                st.markdown("**Top GitHub Repositories:**")
                for repo in repos:
                    st.markdown(f"- {repo}")
                result["actions"].append("▪ ✅ GitHub Search Completed")


    # with col2:
    #     if st.button("🗂️ Create JIRA Task"):
    #         jira_id = create_jira_ticket(st.session_state.last_query)
    #         st.success(f"JIRA Task: {jira_id}")
    #         result["actions"].append(f"▪ ✅ Success — JIRA → {jira_id}")
    # with col2:
    #     jira_create_url = "https://sanjanabathula2003.atlassian.net/jira/software/projects/AI Assistant Demo/issues/?create"
    #     if st.button("🗂️ Create JIRA Task"):
    #         st.markdown(
    #             f"[Click here to open JIRA task creation →]({jira_create_url})",
    #             unsafe_allow_html=True
    #         )
    # with col3:
    #     if st.button("📅 Schedule Meeting"):
    #         meeting_info = schedule_calendar_meeting(st.session_state.last_query)
    #         st.success(f"Meeting Scheduled: {meeting_info}")
    #         result["actions"].append(f"▪ ✅ Success — Calendar → {meeting_info}")
    with col3:
        if st.button("🗂️ Create JIRA Task"):
            with st.spinner("Creating JIRA Task..."):
                try:
                    jira_result = create_jira_ticket(st.session_state.last_query)
                    if isinstance(jira_result, str) and "http" in jira_result:
                        st.success(f"JIRA Task: {jira_result}")
                        result["actions"].append(f"▪ ✅ Success — JIRA → {jira_result}")
                    else:
                        raise ValueError(jira_result)
                except Exception as e:
                    jira_fallback = os.getenv(
                        "JIRA_MANUAL_URL",
                        "https://sanjanabathula2003.atlassian.net/jira/core/projects/AAD/board?groupBy=status"
                    )
                    st.error(f"❌ JIRA task creation failed: {str(e)}")
                    st.markdown(
                        f"[Manually open JIRA board →]({jira_fallback})",
                        unsafe_allow_html=True
                    )

    with col4:
        if st.button("📅 Schedule Meeting"):
            with st.spinner("Scheduling meeting..."):
                try:
                    calendar_result = schedule_calendar_meeting(st.session_state.last_query)
                    if "http" in calendar_result:
                        st.success(f"Meeting Scheduled: {calendar_result}")
                        result["actions"].append(f"▪ ✅ Success — Calendar → {calendar_result}")
                    else:
                        raise ValueError(calendar_result)
                except Exception as e:
                    cal_fallback = os.getenv("CALENDAR_MANUAL_URL", "https://calendar.google.com/calendar/u/0/r/eventedit")
                    st.error(f"❌ Calendar scheduling failed: {str(e)}")
                    st.markdown(f"[Manually schedule meeting →]({cal_fallback})", unsafe_allow_html=True)


    st.subheader("✅ Provide Feedback")

    if not st.session_state.feedback_given:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("✅ Accept"):
                save_reasoning_json(
                result["query"],
                result["role"],
                result["context_chunks"],
                result["trace"],
                result["answer"],
                result["actions"],
                result["reasoning"],
                result["score"],
                result["latency"],
                result["hallucination"],
                "✅",
                result.get("deepeval_passed", False)
            )

                st.session_state.feedback_given = True
                st.success("Feedback recorded: Accepted")
                st.rerun()

        with col_b:
            if st.button("🔄 Retry"):
                new_query = st.session_state.last_query + " (Please rephrase)"
                st.session_state.feedback_given = False
                new_result = ask_question(new_query, st.session_state.last_role.lower())
                st.session_state.last_result = new_result
                st.session_state.last_query = new_query
                st.session_state.chat_history.append((st.session_state.last_role, new_query, new_result))
                st.rerun()

        with col_c:
            if st.button("❌ Reject"):
                new_query = st.session_state.last_query + " (Please refine and improve this answer)"
                st.session_state.feedback_given = False
                new_result = ask_question(new_query, st.session_state.last_role.lower())
                st.session_state.last_result = new_result
                st.session_state.last_query = new_query
                st.session_state.chat_history.append((st.session_state.last_role, new_query, new_result))
                save_reasoning_json(
                    new_result["query"],
                    new_result["role"],
                    new_result["context_chunks"],
                    new_result["trace"],
                    new_result["answer"],
                    new_result["actions"],
                    new_result["reasoning"],
                    new_result["score"],
                    new_result["latency"],
                    new_result["hallucination"],
                    "❌", 
                    new_result.get("deepeval_passed", False)
                )
                st.rerun()
    else:
        st.info("✅ Feedback already submitted. Retry and Reject are disabled.")

if st.session_state.chat_history:
    with st.expander("🗂 Chat History"):
        for role, q, r in reversed(st.session_state.chat_history):
            if r:
                st.markdown(f"**{role}**: {q}")
                st.markdown(f"🧠 {r['answer']}")
                st.markdown("---")
