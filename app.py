from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import logging

from langchain_community.embeddings import FastEmbedEmbeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# -------------------------
# App + ENV setup
# -------------------------
app = Flask(__name__)
load_dotenv()  # locally uses .env; on Render it will use Render env vars

logging.basicConfig(level=logging.INFO)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Fail fast with a clear error (prevents silent None issues)
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
# Embeddings + Vector Store
# -------------------------
# ✅ FastEmbed (no torch / sentence-transformers)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ✅ Make sure this matches the index you created with FastEmbed ingestion
index_name = "medical-chatbot-bge"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# -------------------------
# LLM (Groq)
# -------------------------
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=GROQ_API_KEY
)


# -------------------------
# Prompt
# -------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ]
)


def format_docs(docs):
    """Convert retrieved Document objects into a clean context string."""
    return "\n\n".join(d.page_content for d in docs)


# -------------------------
# RAG Chain (LangChain 1.x)
# -------------------------
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough(),
    }
    | prompt
    | chatModel
    | StrOutputParser()
)


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg.strip():
        return "Please enter a question."

    try:
        logging.info(f"Question: {msg[:200]}")
        response = rag_chain.invoke(msg)
        return response
    except Exception:
        logging.exception("RAG invocation failed")
        return "Sorry — something went wrong on the server. Please try again."


# -------------------------
# Run (local dev only)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
