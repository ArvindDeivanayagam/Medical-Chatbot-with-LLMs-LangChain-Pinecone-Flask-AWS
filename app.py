from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import logging
import time
import re
from functools import lru_cache
from typing import List, Tuple

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.prompt import system_prompt


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
RETRIEVE_K = 12
FINAL_K = 3

CACHE_TTL_SECONDS = 10 * 60
MAX_CACHE_ITEMS = 256


# -------------------------
# Guardrails
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
    r"\bone-sided weakness\b",
    r"\bsuicid(al|e)\b",
    r"\bself-harm\b",
    r"\boverdose\b",
    r"\bheavy bleeding\b",
    r"\bcan'?t stop bleeding\b",
]

MED_ADVICE_PATTERNS = [
    r"\bdosage\b",
    r"\bdose\b",
    r"\bhow much\b",
    r"\bhow often\b",
    r"\bhow many\b",
    r"\bmg\b|\bml\b|\bmilligram\b",
    r"\bshould i take\b",
    r"\bcan i take\b",
    r"\bwhat medicine\b|\bwhich medicine\b",
    r"\bdrug interaction\b|\binteract\b",
    r"\bside effects\b.*\bmed",
]

NON_MEDICAL_RECOMMENDATION_PATTERNS = [
    r"\bbest\b.*\bhospital\b",
    r"\btop\b.*\bhospital\b",
    r"\brecommend\b.*\bhospital\b",
    r"\bnearest\b.*\bhospital\b",
    r"\bnear me\b",
    r"\bin .* hospital\b",
    r"\bwhich hospital\b.*\bgo\b",
]


def guardrail_response(user_text: str) -> str | None:
    t = user_text.lower().strip()

    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, t):
            return (
                "This could be a medical emergency. Please seek immediate medical care "
                "or call your local emergency number right now."
            )

    for pat in NON_MEDICAL_RECOMMENDATION_PATTERNS:
        if re.search(pat, t):
            return "I couldn't find medical information about that in my knowledge base."

    for pat in MED_ADVICE_PATTERNS:
        if re.search(pat, t):
            return (
                "I can’t provide medication dosing or personalized medical advice. "
                "Please consult a licensed clinician or pharmacist for guidance."
            )

    return None


# -------------------------
# Cache
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
# Models
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
    return get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVE_K})


@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=GROQ_API_KEY)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
    ]
)


# -------------------------
# Reranking
# -------------------------
STOPWORDS = {"the","a","an","and","or","to","of","in","on","for","with","is","are","was","were"}

def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def rerank_docs(query: str, docs: List) -> List:
    q_tokens = tokenize(query)
    scored = [(len(q_tokens.intersection(tokenize(d.page_content))), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:FINAL_K]]


def retrieve_and_prepare_context(user_q: str) -> str:
    docs = get_retriever().invoke(user_q)
    docs = rerank_docs(user_q, docs)
    return "\n\n".join(d.page_content for d in docs)


@lru_cache(maxsize=1)
def get_chain():
    context_runnable = RunnableLambda(lambda q: retrieve_and_prepare_context(q))
    return (
        {"context": context_runnable, "input": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    return "ok", 200


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = (request.values.get("msg", "") or "").strip()
    if not msg:
        return "Please enter a question."

    # normalize short queries
    if len(msg.split()) <= 2:
        msg = f"What is {msg.rstrip('?')}?"

    safety = guardrail_response(msg)
    if safety:
        return safety

    cached = cache_get(msg.lower())
    if cached:
        return cached

    try:
        logging.info(f"Incoming question: {msg}")
        response = get_chain().invoke(msg)
        if response.strip():
            cache_set(msg.lower(), response)
        return response
    except Exception as e:
        logging.exception("RAG invocation failed")
        return f"SERVER ERROR: {str(e)}"


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
