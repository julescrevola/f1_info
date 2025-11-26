[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)


# The F1 GPT

Getting information about Formula 1 history with a simple chatbot. Built with LangChain for the backend and React with RSbuild for the frontend.

## Getting started

You will need to install [Node.js](https://nodejs.org/en/download) on your device.

To install react with RS build, you can follow this [tutorial](https://rsbuild.rs/guide/start/quick-start), from which I outline below the main steps.
In your default shell, run:
```powershell
npm create rsbuild@latest
```
After that, run:
```powershell
npm install
npm run dev
```
This will install all dependencies needed and open the React App in your local browser.

## Contributing to the repo

First fork the repo.
You can then clone this repo running:
```bash
git clone https://github.com/julescrevola/f1_info.git
```

### Set up coding environment

To use this repo, first run:
```bash
source cli-aliases.sh
```
This will make sure that the aliases are loaded in your bash terminal.

You can then install the environment with:
```bash
envc
```
And you can update it with:
```bash
envu
```

To install pre-commit hooks, run:
```bash
pre-commit install
```

## Running the local RAG stack

The chatbot is completely local and requires no external API keys. It uses FastF1 for data, DuckDB for storage, HuggingFace embeddings, and Ollama for the LLM.

### 1. Prepare Ollama
```bash
ollama serve   # keep running in a separate terminal
ollama pull mistral
```
You can replace `mistral` with any other model supported by Ollama and set `F1_OLLAMA_MODEL` accordingly.

### 2. Build/update the dataset
Fetch the desired sessions once (defaults to the 2024–2025 seasons):
```bash
python -m src.main  # then call get_data(...) from the REPL, or temporarily uncomment the line in __main__
```
Alternatively, open a Python shell and run:
```python
from src.main import get_data, SESSION_TYPES
get_data(2024, 2025, SESSION_TYPES)
```
This produces `data/all_sessions_results.csv`, which becomes the knowledge base for the chatbot.

### 3. Start the FastAPI backend
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
On startup the server loads the CSV, builds embeddings inside DuckDB, and warms up the RAG chain.

### 4. Start the React frontend
```bash
npm install -g pnpm
cd f1_info_chatbot
pnpm install
pnpm dev
```
Set `VITE_API_URL` in a `.env` file (defaults to `http://localhost:8000`) if the backend runs elsewhere.

### Environment variables
- `F1_START_YEAR` / `F1_END_YEAR`: limit which seasons are ingested into the retriever (defaults 2024–2025).
- `F1_OLLAMA_MODEL`: model name to request from Ollama (`mistral` by default).
- `F1_EMBEDDING_MODEL`: HuggingFace embedding model id (`sentence-transformers/all-MiniLM-L6-v2` by default).
- `VITE_API_URL`: frontend override for the backend URL.
