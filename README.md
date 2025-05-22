# AI-Operating-Officer

# AI Operating Officer

**AI Operating Officer** is an LLM-powered digital executive that mimics enterprise leadership personas (like Satya Nadella, Kevin Scott, Pavan Davuluri) to automate decisions, synthesize strategic recommendations from multimodal sources, and trigger real-world actions like creating Git repos, searching Git repos, JIRA tickets, and meetings.

Built for **next-gen boardroom automation**, this system combines Retrieval-Augmented Generation (RAG), persona-grounded REACT agents, and fallback-aware tool orchestration to create an AI-first Operating Officer that works at the speed of thought.

---

## Features

- **Persona-grounded reasoning** (CEO, CTO, Product Head)
- **Multimodal Retrieval** (PDFs, YouTube captions, LinkedIn blogs)
- **REACT-style Agent Architecture**
- **Tool Automation** (GitHub, JIRA, Microsoft Calendar)
- **Self-healing Fallbacks**
- **RAGAS scoring + Feedback loop**
- **Explainability via JSON traces & footnotes**
- **Streamlit UI with action buttons**

---

## ⚙️ Installation

### 1. Clone the Repo

```bash
git clone https://github.com/Sanjana2903/AI-Operating-Officer.git
cd AI-Operating-Officer
```
### 2. Set up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 3. Configure .env
```bash
OPENAI_API_KEY=your-openai-key
GITLAB_TOKEN=your-gitlab-token
JIRA_URL=https://yourcompany.atlassian.net
JIRA_EMAIL=your-email
JIRA_TOKEN=your-jira-token
MS_GRAPH_TOKEN=your-ms-calendar-token
```
### 4.Ingest Persona Sources

To embed content from transcripts, blogs, or PDFs:
```bash
python ingest.py
```
### 5. Run the app

Using streamlit interface:
```bash
streamlit run app.py
```

## Step-by-Step Breakdown

| Step | Description |
|------|-------------|
| **1. Input** | User enters a natural language query and selects a persona (e.g., Satya as CEO). |
| **2. Retrieval** | Chunks are retrieved from vector DB (Chroma) using cosine similarity. |
| **3. REACT Agent** | The query is passed to a persona-tuned agent prompt that generates a thought process, lifted quotes, and action if needed. |
| **4. Tool Invocation** | If an `Action:` is detected, the system attempts the corresponding automation (GitHub repo, JIRA ticket, calendar event). |
| **5. Fallbacks** | If tool execution fails, fallback URLs are shown (manual GitHub/JIRA/Calendar link). |
| **6. Validation & Scoring** | The full REACT trace is validated. RAGAS scores (F1 + hallucination) are computed. |
| **7. Display Output** | Final answer is shown in the UI with:<br> Paraphrased answer (📘 Lifted Quotes + Footnotes)<br> Suggested Actions (tool links or fallbacks)<br> Agent’s Reasoning |
| **8. Feedback Loop** | User gives feedback (✅ or 🔄 or ❌), saved to reasoning logs for future analysis. |

## Personas Used

| Role     | Sources                                                                 |
|----------|-------------------------------------------------------------------------|
| **CEO**  | Satya Nadella’s Build 2024 Keynote, LinkedIn blogs, YouTube captions    |
| **CTO**  | Kevin Scott’s podcasts, Behind the Tech transcripts                      |
| **Product** | Pavan Davuluri’s Ignite videos, Surface blogposts                    |

---

## Supported Tools

| Tool          | Description                      | Fallback                |
|---------------|----------------------------------|-------------------------|
| **GitHub/GitLab** | Create PoC repositories        | Manual URL              |
| **JIRA**      | Create engineering tickets       | Manual fallback URL     |
| **Calendar**  | Schedule meetings with exec team | Google Calendar link    |

### For video demo: https://drive.google.com/file/d/1n2--Mv0oetwoSwTw8KGXbcha-ZpAxlt8/view?usp=sharing
