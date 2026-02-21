from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import logging
import time
import re
import json
import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Tuple

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.prompt import system_prompt  # keep your system prompt in src/prompt.py


# -------------------------
# App + ENV setup
# -------------------------
app = Flask(__name__)
load_dotenv()

logging.basicConfig(level=logging.INFO)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

missing = []
if not PINECONE_API_KEY:
    missing.append("PINECONE_API_KEY")
if not GROQ_API_KEY:
    missing.append("GROQ_API_KEY")
if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}. "
        f"Set them in Render → Service → Environment."
    )


# -------------------------
# Config
# -------------------------
INDEX_NAME = "medical-chatbot-bge"

# Retrieve more candidates (k=12) then rerank down to top 3
RETRIEVE_K = 12
FINAL_K = 3

# In-memory caching for repeated questions
CACHE_TTL_SECONDS = 10 * 60  # 10 minutes
MAX_CACHE_ITEMS = 256

# Router: allow answering non-medical questions (e.g., Messi)
ALLOW_GENERAL_QA = True
GENERAL_TEMP = 0.4


# -------------------------
# Analytics logging (privacy-safe)
# -------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_question(text: str) -> str:
    # Do NOT store raw user questions (may contain PHI).
    # Store only a short stable hash for analytics.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_event(event: str, **fields) -> None:
    payload = {
        "ts": _now_iso(),
        "event": event,
        **fields,
    }
    # Print is best on Render (shows in Logs)
    print("ANALYTICS " + json.dumps(payload, ensure_ascii=False))


# -------------------------
# Lightweight safety guardrails (rule-based)
# -------------------------
EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bshortness of breath\b",
    r"\bdifficulty breathing\b",
    r"\bfaint(ing)?\b",
    r"\bunconscious\b",
    r"\bseizure\b",
    r"\bstroke\b",
    r"\bface droop\b",
    r"\bslurred speech\b",
    r"\bone[- ]sided weakness\b",
    r"\bsuicid(al|e)\b",
    r"\bself[- ]harm\b",
    r"\boverdose\b",
    r"\bheavy bleeding\b",
    r"\bcan'?t stop bleeding\b",
    r"\bleft arm numb(ness)?\b",
    r"\bnot breathing\b",
]

MED_ADVICE_PATTERNS = [
    # dosing / frequency
    r"\bdosage\b",
    r"\bdose\b",
    r"\bhow much\b.*\bmg\b",
    r"\bhow often\b.*\b(take|use)\b",

    # explicit meds
    r"\bibuprofen\b.*\bhow (often|much)\b",
    r"\bparacetamol\b.*\bhow (often|much)\b",
    r"\bcrocin\b.*\b(dose|how much|how often)\b",

    # prescription / what to take
    r"\bprescription\b",
    r"\bprescribed\b",
    r"\bwhat should i take\b",
    r"\bwhat medicine\b",
    r"\bwhich medicine\b",
    r"\bwhat medication\b",
    r"\bwhich medication\b",
    r"\binhaler\b",
    r"\bantibiotic\b",

    # pregnancy + meds
    r"\bpregnan(t|cy)\b.*\bmed(ic|icine)\b",

    # interactions
    r"\bdrug interaction\b",
    r"\bshould i take\b",
    r"\bcan i take\b",
]


def guardrail_response(user_text: str) -> str | None:
    t = (user_text or "").lower().strip()

    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, t):
            return (
                "This could be a medical emergency. Please seek immediate medical care "
                "or call your local emergency number right now."
            )

    for pat in MED_ADVICE_PATTERNS:
        if re.search(pat, t):
            return (
                "I can’t provide medication dosing or personalized medical advice. "
                "Please consult a licensed clinician or pharmacist for guidance."
            )

    return None


# -------------------------
# Caching helpers (simple TTL cache)
# -------------------------
_answer_cache: dict[str, Tuple[float, str]] = {}


def cache_get(key: str) -> str | None:
    item = _answer_cache.get(key)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > CACHE_TTL_SECONDS:
        _answer_cache.pop(key, None)
        return None
    return val


def cache_set(key: str, val: str) -> None:
    if len(_answer_cache) >= MAX_CACHE_ITEMS:
        oldest_key = min(_answer_cache.items(), key=lambda kv: kv[1][0])[0]
        _answer_cache.pop(oldest_key, None)
    _answer_cache[key] = (time.time(), val)


# -------------------------
# Router helper (medical vs non-medical)
# -------------------------
MEDICAL_KEYWORDS = [
    "symptom", "symptoms", "pain", "fever", "cough", "wheezing", "breath", "asthma",
    "fracture", "infection", "virus", "bacteria", "disease", "medicine", "medication",
    "tablet", "dose", "hospital", "clinic", "diagnosis", "treatment", "side effect",
    "blood", "pressure", "diabetes", "cancer", "heart", "chest", "stomach",
    "headache", "vomit", "nausea", "rash", "injury", "bone", "osteoporosis"
]


def is_medical_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in MEDICAL_KEYWORDS)


# -------------------------
# Model + vectorstore (cached in memory)
# -------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def get_vectorstore():
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVE_K},
    )


@lru_cache(maxsize=1)
def get_llm():
    # medical RAG LLM
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )


@lru_cache(maxsize=1)
def get_general_llm():
    # non-medical general QA LLM
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=GENERAL_TEMP,
        api_key=GROQ_API_KEY,
    )


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
    ]
)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# -------------------------
# Lightweight reranking (keyword overlap baseline)
# -------------------------
STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "be","been","being","as","at","by","from","that","this","it","you","your","i","we",
    "they","them","their","but","not","do","does","did","can","could","should","would"
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def rerank_docs(query: str, docs: List) -> List:
    q_tokens = tokenize(query)
    if not q_tokens:
        return docs[:FINAL_K]

    scored = []
    for d in docs:
        d_tokens = tokenize(getattr(d, "page_content", ""))
        score = len(q_tokens.intersection(d_tokens))
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:FINAL_K]]


def retrieve_and_prepare_context(user_q: str) -> Tuple[str, int, int]:
    """
    Returns: (context_text, doc_count_before_rerank, doc_count_after_rerank)
    """
    retriever = get_retriever()
    docs = retriever.invoke(user_q)  # List[Document]
    before = len(docs)
    docs = rerank_docs(user_q, docs)
    after = len(docs)
    return format_docs(docs), before, after


@lru_cache(maxsize=1)
def get_chain():
    """
    Chain takes a dict input: {"context": "...", "input": "..."} to avoid double retrieval.
    """
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain


def general_answer(user_q: str) -> str:
    llm = get_general_llm()
    p = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Keep answers concise."),
            ("human", "{q}"),
        ]
    )
    chain = p | llm | StrOutputParser()
    return chain.invoke({"q": user_q})


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    return "ok", 200


@app.route("/warmup")
def warmup():
    return "warmed", 200


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.values.get("msg", "")
    msg = (msg or "").strip()

    if not msg:
        return "Please enter a question."

    q_hash = _hash_question(msg)
    t0 = time.perf_counter()

    # 1) Guardrails first
    safety = guardrail_response(msg)
    if safety:
        log_event(
            "request",
            q_hash=q_hash,
            route="guardrail",
            outcome="guardrail",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return safety

    # 2) Cache next
    cache_key = msg.lower()
    cached = cache_get(cache_key)
    if cached:
        log_event(
            "request",
            q_hash=q_hash,
            route="cache",
            outcome="cache_hit",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return cached

    try:
        logging.info(f"Incoming question: {msg[:200]}")

        # Decide route
        use_general = ALLOW_GENERAL_QA and (not is_medical_query(msg))

        if use_general:
            # General QA (no Pinecone)
            t_llm = time.perf_counter()
            response = general_answer(msg)
            llm_elapsed = time.perf_counter() - t_llm
            total_elapsed = time.perf_counter() - t0

            if isinstance(response, str) and response.strip():
                cache_set(cache_key, response)

            log_event(
                "request",
                q_hash=q_hash,
                route="general",
                outcome="ok",
                retrieval_latency_ms=0,
                llm_latency_ms=int(llm_elapsed * 1000),
                latency_ms=int(total_elapsed * 1000),
            )
            return response

        # Medical RAG path
        t_retr = time.perf_counter()
        context_text, doc_before, doc_after = retrieve_and_prepare_context(msg)
        retr_elapsed = time.perf_counter() - t_retr

        chain = get_chain()
        t_llm = time.perf_counter()
        response = chain.invoke({"context": context_text, "input": msg})
        llm_elapsed = time.perf_counter() - t_llm

        total_elapsed = time.perf_counter() - t0
        logging.info(f"Latency_seconds={total_elapsed:.2f}")

        if isinstance(response, str) and response.strip():
            cache_set(cache_key, response)

        log_event(
            "request",
            q_hash=q_hash,
            route="rag",
            outcome="ok",
            doc_count=doc_before,
            final_k=doc_after,
            retrieval_latency_ms=int(retr_elapsed * 1000),
            llm_latency_ms=int(llm_elapsed * 1000),
            latency_ms=int(total_elapsed * 1000),
        )

        return response

    except Exception as e:
        total_elapsed = time.perf_counter() - t0
        log_event(
            "request",
            q_hash=q_hash,
            route="error",
            outcome="error",
            error=str(e)[:200],
            latency_ms=int(total_elapsed * 1000),
        )
        logging.exception("RAG invocation failed")
        return f"SERVER ERROR: {str(e)}"


# -------------------------
# Run (local dev only)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)