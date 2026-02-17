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

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise RuntimeError("Missing required environment variables.")


# -------------------------
# Config
# -------------------------
INDEX_NAME = "medical-chatbot-bge"
RETRIEVE_K = 12
FINAL_K = 3
CACHE_TTL_SECONDS = 600
MAX_CACHE_ITEMS = 256


# -------------------------
# Guardrails
# -------------------------
EMERGENCY_PATTERNS = [
    r"\bchest pain\b", r"\bshortness of breath\b", r"\bfaint(ing)?\b",
    r"\bunconscious\b", r"\bseizure\b", r"\bstroke\b",
    r"\bslurred speech\b", r"\bone-sided weakness\b", r"\boverdose\b"
]

MED_ADVICE_PATTERNS = [
    r"\bdosage\b", r"\bdose\b", r"\bhow much\b.*\bmg\b",
    r"\bshould I take\b", r"\bcan I take\b", r"\bdrug interaction\b"
]


def guardrail_response(text: str):
    t = text.lower()

    for p in EMERGENCY_PATTERNS:
        if re.search(p, t):
            return "This could be a medical emergency. Please seek immediate medical care or call your local emergency number right now."

    for p in MED_ADVICE_PATTERNS:
        if re.search(p, t):
            return "I can’t provide medication dosing or personalized medical advice. Please consult a licensed clinician or pharmacist for guidance."

    return None


# -------------------------
# Cache
# -------------------------
_answer_cache: dict[str, Tuple[float, str]] = {}

def cache_get(key):
    v = _answer_cache.get(key)
    if not v:
        return None
    ts, val = v
    if time.time() - ts > CACHE_TTL_SECONDS:
        _answer_cache.pop(key, None)
        return None
    return val

def cache_set(key, val):
    if len(_answer_cache) >= MAX_CACHE_ITEMS:
        oldest = min(_answer_cache.items(), key=lambda x: x[1][0])[0]
        _answer_cache.pop(oldest, None)
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
        embedding=get_embeddings()
    )

@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVE_K})

@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=GROQ_API_KEY)


# -------------------------
# Reranking
# -------------------------
STOPWORDS = {"the","a","an","and","or","to","of","in","on","for","with","is","are","was","were"}

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w)>2}

def rerank_docs(query, docs):
    q = tokenize(query)
    scored=[]
    for d in docs:
        score=len(q.intersection(tokenize(d.page_content)))
        scored.append((score,d))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [d for _,d in scored[:FINAL_K]]

def build_context(q):
    docs=get_retriever().invoke(q)
    docs=rerank_docs(q,docs)
    return "\n\n".join(d.page_content for d in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])


@lru_cache(maxsize=1)
def get_chain():
    return (
        {"context": RunnableLambda(build_context), "input": RunnablePassthrough()}
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


@app.route("/warmup")
def warmup():
    try:
        logging.info("Warmup started")
        _ = get_embeddings()
        _ = get_vectorstore()
        _ = get_retriever()
        chain = get_chain()
        chain.invoke("hello")
        logging.info("Warmup complete")
        return "warmed", 200
    except Exception as e:
        logging.exception("Warmup failed")
        return f"warmup failed: {str(e)}", 500


@app.route("/get", methods=["GET","POST"])
def chat():
    msg=(request.values.get("msg","") or "").strip()
    if not msg:
        return "Please enter a question."

    safety=guardrail_response(msg)
    if safety:
        return safety

    cached=cache_get(msg.lower())
    if cached:
        return cached

    try:
        response=get_chain().invoke(msg)
        if isinstance(response,str) and response.strip():
            cache_set(msg.lower(),response)
        return response
    except Exception as e:
        logging.exception("RAG invocation failed")
        return f"SERVER ERROR: {str(e)}"


# -------------------------
# Run local
# -------------------------
if __name__=="__main__":
    port=int(os.environ.get("PORT",8080))
    app.run(host="0.0.0.0",port=port)
