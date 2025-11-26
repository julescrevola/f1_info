import os
from functools import lru_cache
from pathlib import Path
from typing import List

import duckdb
import fastf1 as ff1
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.vectorstores import DuckDB
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from tqdm import tqdm

# ---------------------------------------------------------------------------#
# Paths & constants
# ---------------------------------------------------------------------------#
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_PATH = DATA_DIR / "all_sessions_results.csv"
VECTOR_DB_PATH = DATA_DIR / "f1.duckdb"
VECTOR_TABLE = "f1_embeddings"
FASTF1_CACHE = BASE_DIR / "fastf1_cache"
DOCUMENT_COUNT = 0

SESSION_TYPES = ["Q", "S", "SQ", "SS", "R"]
DEFAULT_START_YEAR = int(os.getenv("F1_START_YEAR", "2024"))
DEFAULT_END_YEAR = int(os.getenv("F1_END_YEAR", "2025"))

# Ensure directories exist and FastF1 cache is ready
DATA_DIR.mkdir(exist_ok=True)
FASTF1_CACHE.mkdir(exist_ok=True)
ff1.Cache.enable_cache(str(FASTF1_CACHE))

# ---------------------------------------------------------------------------#
# FastAPI application setup
# ---------------------------------------------------------------------------#
app = FastAPI(title="F1 RAG Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------#
# Data collection utilities
# ---------------------------------------------------------------------------#
def get_data(start_year: int, end_year: int, session_types: List[str]) -> pd.DataFrame:
    """Download F1 session data for given years and session types."""
    all_results = []

    for year in range(start_year, end_year + 1):
        try:
            if year <= 2018:
                schedule = ff1.get_event_schedule(
                    year, include_testing=False, backend="ergast"
                )
            else:
                schedule = ff1.get_event_schedule(year, include_testing=False)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Could not load schedule for {year}: {exc}")
            continue

        for _, event in tqdm(schedule.iterrows(), total=len(schedule)):
            event_name = event["EventName"]
            round_num = event["RoundNumber"]
            print(f"\nProcessing {year} {event_name} (Round {round_num})")

            for session_type in session_types:
                try:
                    session = ff1.get_session(year, event_name, session_type)
                    session.load(laps=True, telemetry=True, weather=True, messages=True)

                    results = session.results
                    if results is not None:
                        results["Year"] = year
                        results["Event"] = event_name
                        results["Round"] = round_num
                        results["Session"] = session_type
                        all_results.append(results)

                    laps = session.laps
                    if laps is not None and not laps.empty:
                        laps["Year"] = year
                        laps["Event"] = event_name
                        laps["Round"] = round_num
                        laps["Session"] = session_type
                        all_results.append(laps)

                except Exception as exc:  # pylint: disable=broad-except
                    print(f"Skipped {year} {event_name} {session_type}: {exc}")
                    continue

    if not all_results:
        raise RuntimeError("No session results collected.")

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Saved combined results to {RESULTS_PATH}")
    return combined_results


# ---------------------------------------------------------------------------#
# RAG building blocks
# ---------------------------------------------------------------------------#
def _row_to_text(row: dict) -> str:
    """Convert a results row into a concise text snippet."""
    important_fields = [
        "Year",
        "Event",
        "Round",
        "Session",
        "DriverNumber",
        "FullName",
        "TeamName",
        "Position",
        "Time",
        "Status",
        "FastestLapTime",
        "Stint",
        "Compound",
    ]
    parts = [
        f"{field}: {row[field]}"
        for field in important_fields
        if field in row and row[field] not in ("", None)
    ]
    return " | ".join(parts)


def load_documents() -> List[Document]:
    """Load CSV data and transform each row into a LangChain Document."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"No dataset found at {RESULTS_PATH}. Run get_data() first to build the knowledge base."
        )

    df = pd.read_csv(RESULTS_PATH).fillna("")

    if "Year" in df.columns:
        df = df[(df["Year"] >= DEFAULT_START_YEAR) & (df["Year"] <= DEFAULT_END_YEAR)]

    records = df.to_dict(orient="records")
    documents = []

    for row in records:
        metadata = {
            "year": row.get("Year"),
            "event": row.get("Event"),
            "session": row.get("Session"),
            "driver": row.get("FullName") or row.get("DriverNumber"),
            "team": row.get("TeamName"),
        }
        documents.append(
            Document(
                page_content=_row_to_text(row),
                metadata={k: v for k, v in metadata.items() if v},
            )
        )

    if not documents:
        raise RuntimeError(
            "No documents were created from the dataset. Check filtering criteria."
        )

    return documents


def _clear_vector_table() -> None:
    """Ensure the DuckDB table starts clean before re-ingesting documents."""
    if not VECTOR_DB_PATH.exists():
        return

    with duckdb.connect(str(VECTOR_DB_PATH)) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {VECTOR_TABLE}")


@lru_cache(maxsize=1)
def get_embeddings():
    model_name = os.getenv(
        "F1_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=1)
def get_vector_store() -> DuckDB:
    global DOCUMENT_COUNT  # pylint: disable=global-statement
    documents = load_documents()
    _clear_vector_table()
    DOCUMENT_COUNT = len(documents)
    return DuckDB.from_documents(
        documents,
        embedding=get_embeddings(),
        table=VECTOR_TABLE,
        database=str(VECTOR_DB_PATH),
    )


def format_docs(docs: List[Document]) -> str:
    """Format retrieved docs for the prompt."""
    return "\n---\n".join(
        f"{doc.metadata.get('year', '?')} {doc.metadata.get('event', '')} {doc.metadata.get('session', '')}\n{doc.page_content}"
        for doc in docs
    )


@lru_cache(maxsize=1)
def get_rag_chain():
    retriever = get_vector_store().as_retriever(
        search_kwargs={"k": 3}
    )  # Reduced from 4 to 3 for faster retrieval
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert F1 analyst. Answer concisely using ONLY the provided context. "
                "If information is missing, say so. Be brief and factual.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
        ]
    )
    llm = Ollama(
        model=os.getenv("F1_OLLAMA_MODEL", "mistral"),
        temperature=0.1,  # Low temperature for faster, more deterministic responses
    )
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )


# ---------------------------------------------------------------------------#
# API routes
# ---------------------------------------------------------------------------#
@app.on_event("startup")
async def warm_up_chain() -> None:
    """Eagerly build the retriever/LLM pipeline so first request is fast."""
    import asyncio

    print("🔥 Starting F1 RAG Chatbot backend...")
    print("🚀 Server ready! API available at http://localhost:8000")
    print("📚 Loading documents and building RAG chain in background...")

    async def warmup_task():
        """Run warmup in background so server can respond immediately."""
        try:
            # Run blocking operation in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, get_rag_chain)
            print(f"✅ RAG chain ready! Loaded {DOCUMENT_COUNT} documents.")
        except FileNotFoundError as exc:
            print(f"⚠️  RAG warm-up failed: {exc}")
            print(
                "   The server is running, but chat requests will fail until data is available."
            )
        except Exception as exc:  # pylint: disable=broad-except
            import traceback

            print(f"⚠️  RAG warm-up failed: {exc}")
            print("   The server is running, but chat requests may fail.")
            traceback.print_exc()

    # Start warmup in background
    asyncio.create_task(warmup_task())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": DOCUMENT_COUNT}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    import asyncio

    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        chain = get_rag_chain()
        # Run the RAG chain in an executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, chain.invoke, question)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset not found. Run get_data() to download FastF1 results and restart the API. "
                f"Missing file: {RESULTS_PATH}"
            ),
        ) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500, detail=f"Unable to answer question: {exc}"
        ) from exc

    if not isinstance(answer, str):
        answer = str(answer)

    return ChatResponse(response=answer)


# ---------------------------------------------------------------------------#
# Entry point helpers (manual runs)
# ---------------------------------------------------------------------------#
def test_llm(model: str = "mistral") -> None:
    llm = Ollama(model=model)
    response = llm.invoke("Give me a short fact about Formula 1.")
    if response:
        print("✅ LLM is working.")
    else:
        print("❌ LLM is not working.")


if __name__ == "__main__":
    # Example usage for manual data refresh & quick test
    # get_data(DEFAULT_START_YEAR, DEFAULT_END_YEAR, SESSION_TYPES)
    print("Starting F1 RAG Chatbot backend...")
