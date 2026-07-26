# 🤖 AI Executive Assistant --- Backend

A tool-using **Agentic AI Executive Assistant** built with **FastAPI,
LangGraph, LangChain, MCP (Model Context Protocol), and Google Workspace
APIs**.

Instead of behaving like a normal chatbot that only generates text, this
assistant can **take actions**: work with Gmail and Google Calendar,
create Google Meet spaces, search the web, check weather, convert
currencies, generate PDF/Excel files, attach generated files to emails,
and pause sensitive actions for **Human-in-the-Loop (HITL)** approval.

> Frontend repository:
> https://github.com/sarfraj0304/AI-Executive-Assistant-Frontend

------------------------------------------------------------------------

## ✨ Key Features

-   🧠 LangGraph agent workflow with LangChain tool calling
-   🔌 MCP server built with FastMCP and loaded through
    `MultiServerMCPClient`
-   📧 Gmail: read emails, send emails, multiple recipients, real file
    attachments
-   📅 Google Calendar: read, create, update and delete events
-   🎥 Google Meet REST API integration
-   🌐 Tavily web search
-   🌦️ Live weather tool
-   💱 Currency conversion and calculation tools
-   📄 PDF generation with ReportLab
-   📊 Excel generation with OpenPyXL
-   🛡️ Generic Human-in-the-Loop approval for sensitive tools
-   💬 Approval/edit/reject using normal chat text instead of a popup
-   🧵 Thread-based LangGraph conversations
-   💾 Development checkpointing with `InMemorySaver`
-   🔗 Multi-tool workflows such as Calendar → Excel → Gmail

## 🧠 Why this is Agentic AI

A normal chatbot mainly returns text. This application can choose tools,
execute them, observe their results and continue with another tool.

``` text
User:
"Get tomorrow's meetings, convert them to Excel,
and email the file to Prakash."

        ↓
   LangGraph Agent
        ↓
Google Calendar Tool
        ↓
   Excel Export Tool
        ↓
    Gmail Tool
        ↓
HITL Approval in Chat
        ↓
      User: yes
        ↓
Email sent with the actual .xlsx attachment
```

Another workflow:

``` text
"Explain Agentic AI, create a PDF and email it."

LLM → PDF Tool → Gmail Tool → Approval → 📎 PDF Attachment
```

## 🛡️ Human-in-the-Loop

Sensitive operations can pause before execution. Approval is shown
directly in the conversation.

``` text
Assistant:
Do you want to send this email?

To: user@example.com
Subject: Meeting Report
Attachment: meeting_report.pdf

User:
yes
```

The user can also respond with requests such as:

``` text
no
cancel
change the subject to "Tomorrow's Schedule"
rewrite the body more professionally
```

Approval-required tools are centrally configured, so you do not need a
separate LangGraph approval node for every sensitive tool.

## 📧 Gmail

The Gmail integration supports the permissions needed for reading,
composing, sending and modifying email.

Capabilities include:

-   Read/retrieve email information
-   Send email
-   Multiple recipients
-   Generated PDF/Excel attachments
-   MIME attachments rather than merely inserting a file URL
-   HITL before configured sensitive actions

For generated files, the recipient receives a real attachment:

``` text
📎 report.pdf
📎 meetings.xlsx
```

## 📅 Google Calendar

The Calendar integration supports workflows such as:

-   Retrieve events
-   View upcoming meetings
-   Create events
-   Update events
-   Delete events
-   Pass calendar data to PDF/Excel tools
-   Combine Calendar + Gmail in a single user request

## 🎥 Google Meet

Google Meet REST API integration enables Meet-related workflows such as
creating/accessing meeting spaces and combining Meet information with
Calendar or Gmail.

> Availability of advanced Meet artifacts such as transcripts,
> recordings or participant data depends on Google Workspace permissions
> and whether that artifact exists for the conference.

## 📄 PDF & 📊 Excel

Text can be exported to PDF and structured data can be exported to
Excel.

Exports conceptually return:

``` json
{
  "success": true,
  "file_name": "unique_report.pdf",
  "file_url": "http://127.0.0.1:8000/exports/unique_report.pdf"
}
```

Use them differently:

``` text
file_name → internal tool-to-tool use / real Gmail attachment
file_url  → browser download
```

## 🏗️ Architecture

``` text
┌──────────────────────────┐
│      React Frontend      │
└────────────┬─────────────┘
             │ HTTP
             ▼
┌──────────────────────────┐
│         FastAPI          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        LangGraph         │
│     Agent ↔ ToolNode     │
│            │             │
│       HITL Interrupt     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        MCP Server        │
├──────────────────────────┤
│ Gmail                    │
│ Calendar                 │
│ Google Meet              │
│ Tavily Search            │
│ Weather / Currency       │
│ PDF / Excel Export       │
│ Additional MCP Tools     │
└──────────────────────────┘
```

## 🛠️ Tech Stack

  Technology                Purpose
  ------------------------- ------------------------------
  Python                    Runtime
  FastAPI                   REST API
  LangGraph                 Agent workflow/orchestration
  LangChain                 LLM and tool integration
  MCP / FastMCP             Tool server
  OpenRouter / ChatOpenAI   LLM access
  Gmail API                 Email automation
  Google Calendar API       Calendar automation
  Google Meet REST API      Meet integration
  Tavily                    Web search
  ReportLab                 PDF generation
  OpenPyXL                  Excel generation
  HTTPX                     HTTP requests

## 📁 Project Structure

``` text
Backend/
├── app/
│   ├── main.py
│   ├── routes.py
│   ├── graph.py
│   ├── state.py
│   ├── models.py
│   ├── config.py
│   ├── prompts.py
│   ├── mcp_server.py
│   ├── tools/
│   │   ├── gmail.py
│   │   ├── calender.py
│   │   ├── meet.py
│   │   ├── export.py
│   │   ├── search.py
│   │   └── whatsapp.py
│   └── utils/
│       ├── approval/
│       │   ├── approval.py
│       │   └── approval_tools.py
│       └── format_messages.py
├── exports/
├── credentials.json       # create locally — DO NOT COMMIT
├── token.json             # generated locally — DO NOT COMMIT
├── requirements.txt
└── .env                   # DO NOT COMMIT
```

# ⚡ Installation

## 1. Clone

``` bash
git clone https://github.com/sarfraj0304/AI-Executive-Assistant-Backend.git
cd AI-Executive-Assistant-Backend
```

## 2. Virtual environment

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Dependencies

``` bash
pip install -r requirements.txt
```

# 🔐 Environment Variables

Create `.env` in the backend root.

``` env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENAI_API_KEY=your_openai_api_key_if_used
OPENWEATHER_API_KEY=your_openweather_api_key
TAVILY_API_KEY=your_tavily_api_key
BASE_URL=http://127.0.0.1:8000
```

Only configure keys for services enabled in your local version.

API providers:

-   OpenRouter: https://openrouter.ai/
-   OpenAI: https://platform.openai.com/
-   Tavily: https://tavily.com/
-   OpenWeather: https://openweathermap.org/api

# 🔑 Google Workspace Setup

Each developer/user should create their own Google Cloud OAuth
credentials.

## 1. Create a Google Cloud project

Open:

https://console.cloud.google.com/

Create/select a project.

## 2. Enable Google APIs

From **APIs & Services → Library**, enable:

1.  **Gmail API**
2.  **Google Calendar API**
3.  **Google Meet REST API**

## 3. Configure OAuth consent

Open the Google Auth Platform / OAuth consent configuration and complete
the application details.

For local development, the app can remain in **Testing** mode.

If the OAuth app is in Testing mode, add the Google accounts that should
be able to authenticate as **test users**.

## 4. Create OAuth Client ID

Go to:

``` text
APIs & Services
→ Credentials
→ Create Credentials
→ OAuth client ID
```

For the local `InstalledAppFlow`, create a **Desktop app** OAuth client.

Download the JSON file.

## 5. Create `credentials.json`

Rename the downloaded file:

``` text
credentials.json
```

Place it in the backend root:

``` text
AI-Executive-Assistant-Backend/
├── app/
├── credentials.json
├── requirements.txt
└── ...
```

Never commit this file.

## 6. Generate `token.json`

Activate your virtual environment and run this command from the backend
root:

``` bash
python -c "from app.tools.gmail import get_gmail_service; get_gmail_service(); print('Auth successful, token.json created')"
```

A browser window should open.

1.  Choose your Google account.
2.  Review permissions.
3.  Approve access.
4.  Return to the terminal.

On success:

``` text
Auth successful, token.json created
```

You should now have:

``` text
credentials.json
token.json
```

### Changed OAuth scopes?

If Gmail/Calendar/Meet scopes change, the existing token may not contain
the new permissions.

Delete:

``` text
token.json
```

Then authenticate again:

``` bash
python -c "from app.tools.gmail import get_gmail_service; get_gmail_service(); print('Auth successful, token.json created')"
```

# ▶️ Run

``` bash
uvicorn app.main:app --reload
```

Backend:

``` text
http://127.0.0.1:8000
```

FastAPI docs:

``` text
http://127.0.0.1:8000/docs
```

# 💬 Example Prompts

``` text
What's the weather in Delhi?

Show my latest emails.

Show my meetings tomorrow.

Create a Google Meet.

Explain Agentic AI and convert it into PDF.

Get my meetings this week and convert them into Excel.

Explain Agentic AI, create a PDF and email the PDF as an attachment.

Get tomorrow's meetings, create an Excel file and email the Excel file.
```

# 🧵 Conversation Threads

The API uses `thread_id` so LangGraph can maintain separate conversation
state.

Conceptually:

``` json
{
  "thread_id": "1",
  "message": "Show my recent emails"
}
```

Reuse the same thread ID to continue a conversation.

> `InMemorySaver` is suitable for development. Persistent checkpoint
> storage is recommended for production.

# 🛡️ HITL Flow

``` text
Tool requested
     ↓
Approval required?
     ↓ yes
LangGraph interrupt
     ↓
Frontend renders approval in chat
     ↓
User: yes / no / requested change
     ↓
Graph resumes
```

# 🔒 Security

Your `.gitignore` should include:

``` gitignore
.env
credentials.json
token.json
exports/
venv/
.venv/
__pycache__/
*.pyc
```

## 🚨 Important if secrets were pushed to GitHub

Removing `.env`, `credentials.json`, or `token.json` in a later commit
is not enough if they existed in Git history.

If any were committed:

1.  Rotate exposed API keys.
2.  Revoke affected Google OAuth tokens.
3.  Create fresh credentials/tokens where needed.
4.  Remove secrets from Git history if the repository is public.

# 🧩 Adding New MCP Tools

Add a function decorated with `@mcp.tool()` in the MCP server and give
it a clear description so the model knows when it should be used.

If the action is sensitive, add its tool name to the centralized
approval configuration.

# 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'app...'`

Run commands from the backend root:

``` bash
uvicorn app.main:app --reload
```

### MCP `Connection closed`

Check that the MCP server starts successfully, imports use valid `app.*`
paths, dependencies are installed, and the subprocess uses the expected
Python environment.

### Google `invalid_scope`

Check the configured OAuth scopes, delete `token.json`, and authenticate
again.

### Calendar/Meet API not enabled

Enable the API in the same Google Cloud project that created
`credentials.json`.

### Attachment not found

The Gmail tool should receive the generated `file_name`, not the browser
URL:

``` text
✅ unique_report.pdf
❌ http://127.0.0.1:8000/exports/unique_report.pdf
```

# 🌱 Future Improvements

-   Persistent LangGraph checkpoint storage
-   Database-backed users and threads
-   Per-user Google OAuth tokens
-   Streaming responses
-   Concise long-term conversation summaries
-   Structured planning/replanning for complex tasks
-   More Google Meet artifact tools
-   Background/scheduled agent tasks
-   Production authentication/authorization
-   Additional MCP integrations

# 🤝 Contributing

Contributions and suggestions are welcome.

``` bash
git checkout -b feature/my-feature
git commit -m "feat: add my feature"
git push origin feature/my-feature
```

Then open a Pull Request.

## 🔗 Repositories

-   Backend:
    https://github.com/sarfraj0304/AI-Executive-Assistant-Backend
-   Frontend:
    https://github.com/sarfraj0304/AI-Executive-Assistant-Frontend

## ⭐ Support

If this project helps you learn or build Agentic AI systems, consider
starring the repositories.

Built as a practical demonstration of **LangGraph + MCP + HITL +
real-world tool execution**.
