# Gemini Chat - WebSocket Application

A real-time chat application using FastAPI, WebSocket, and Google's Gemini 1.5 Pro model via LangChain.

## 🚀 Features

- Real-time streaming responses from Gemini 1.5 Pro
- WebSocket-based communication
- Beautiful, modern chat interface
- Conversation memory
- Auto-reconnection on disconnect

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install fastapi uvicorn langchain-google-genai google-generativeai langchain websockets
```

### 2. Setup Environment

Set your Google API Key as an environment variable:

**On macOS/Linux:**
```bash
export GOOGLE_API_KEY=your_api_key_here
```

**On Windows:**
```bash
set GOOGLE_API_KEY=your_api_key_here
```

Or create a `.env` file (requires `python-dotenv`):
```
GOOGLE_API_KEY=your_api_key_here
POSTGRES_URL=postgresql://bigstep@localhost:5432/langchain_memory
```

### 3. Configure Postgres short-term memory (async)

1. Make sure PostgreSQL is installed (e.g., `brew install postgresql@14`) and the service is running:
   ```bash
   brew services start postgresql@14
   ```
2. Create the database the app will use for LangGraph checkpoints:
   ```bash
   createdb langchain_memory
   ```
3. Set `POSTGRES_URL` (or add it to `.env`) so `main.py` can connect. Example:
   ```bash
   export POSTGRES_URL=postgresql://$(whoami)@localhost:5432/langchain_memory
   ```
   The application falls back to this local URL automatically if the variable is missing.

> The server now uses LangGraph's `AsyncPostgresSaver`, so the dependency list includes both `psycopg[binary]` (for CLI access) and `asyncpg` (for the async connection pool). Running `pip install -r requirements.txt` installs everything needed.

## 🏃 Running the Application

**Important:** Make sure your virtual environment is activated first!

Start the server:

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate

# Run using python -m uvicorn (recommended - uses venv Python)
python -m uvicorn main:app --reload --port 8000
```

Or run directly:

```bash
python main.py
```

**Note:** If you get `ModuleNotFoundError`, make sure:
1. Virtual environment is activated (you should see `(.venv)` in your prompt)
2. Use `python -m uvicorn` instead of just `uvicorn` to ensure it uses the venv's Python

Then open your browser and navigate to:
```
http://localhost:8000
```

## 📁 Project Structure

```
langchain-training/
├── main.py              # FastAPI application with WebSocket endpoint
├── index.html           # Chat interface HTML page
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Configuration

You can modify the Gemini model settings in `main.py`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Change model here
    streaming=True,
    temperature=0.2          # Adjust creativity (0.0-1.0)
)
```

## 🎯 Usage

1. Start the server
2. Open `http://localhost:8000` in your browser
3. Wait for "Connected" status
4. Type your message and press Enter or click Send
5. Watch the AI response stream in real-time!

## 🛠️ Troubleshooting

- **Connection issues**: Make sure port 8000 is not in use
- **API Key errors**: Verify your `GOOGLE_API_KEY` is set correctly
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

## 📝 Notes

- The application uses conversation memory, so context is maintained across messages
- WebSocket automatically reconnects if the connection is lost
- Responses stream token-by-token for a better user experience

